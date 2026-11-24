#!/usr/bin/env python
"""Per-hour aerosol composition for the PARSEC inputs, from observations where possible and CAMS otherwise.

For every hour in inputs/<site>_parcel_inputs.nc this writes the dry mass fractions (massfrac, PARSEC
order sites.SPECIES), the absolute masses behind them and a comp_source flag, following the hierarchy

  acsm      hourly ACSM (organics, sulfate, ammonium, nitrate, chloride; ARM b2 > b1, or EBAS),
            measured eBC if there is any, else CAMS BC; CAMS sea salt and dust (submicron bins)
  filter    daily EBAS filter-pack ions: non-sea-salt sulfate and ammonium; sea salt = 3.26 x Na,
            scaled to its submicron share with CAMS' sea-salt bins; organics = CAMS OM/SO4 x measured
            sulfate; measured eBC if any, else CAMS BC; CAMS dust
  ccn       (only with --ccn-tier) hours that would be "bc" or "cams" but have a valid kappa from the
            CCN counter (build_ccn_kappa.py): the (nh4)2so4 + organics volume is re-split so the bulk
            kappa matches kappa_ccn at the setpoint nearest --ccn-ss; BC, dust and sea salt unchanged
  bc        measured eBC, everything else CAMS
  cams      CAMS only

and appends where the CAMS part came from: "+cams3h" (3-hourly EAC4, nearest time within 2 h),
"+camsmon" (EAC4 monthly mean) or "+camsclim" (mean of the same calendar month in other years, for
hours after EAC4 ends). E.g. "acsm+bc_obs+cams3h", "filter+bc_cams+camsmon", "cams+camsclim".

Species mapping (PARSEC has no nitrate or chloride):
  sulfate      split between h2so4 / nh4hso4 / (nh4)2so4 from the NH4:SO4 molar ratio R
               (R>=2 all (NH4)2SO4; 1<=R<2 (NH4)HSO4 + (NH4)2SO4; R<1 H2SO4 + (NH4)HSO4). Sulfate takes
               ammonium first. Without measured ammonium (CAMS, some filters) sulfate is taken as
               fully neutralised (NH4)2SO4.
  nitrate, chloride   added to the (nh4)2so4 column as NH4NO3 / NH4Cl mass (kappa ~0.6-0.7, close
               to ammonium sulfate). Only from the ACSM: filter nitrate is HNO3 + NO3 or coarse.
  organics     oc column (organic matter mass, not carbon); svoc stays 0
  eBC          absorption / MAC, MAC = 6.6 m2 g-1 at 637 nm scaled as 1/lambda (factor ~2 uncertain;
               BC is a few percent of the mass, so this hardly matters for activation)
  sea salt     CAMS 0.03-0.5 um bin (radius at 80 % RH, i.e. roughly dry diameter < 0.5 um) / 4.3
               for the dry mass; dust the CAMS 0.03-0.55 um bin

CAMS (kg kg-1) is taken at the grid point nearest the site and the pressure level nearest the
station's standard-atmosphere pressure, and converted to ug m-3 with the standard-atmosphere density
there. A bulk kappa (volume-weighted, with the kappa PARSEC's own Koehler curve implies for each column) is
stored as a diagnostic.

The placeholder fractions are kept as massfrac_placeholder the first time the file is updated.
Hours are taken as the start of the hour, so the matching window is [hour, hour + 1 h).

Examples
--------
python build_composition.py                        # all sites with an inputs file
python build_composition.py --sites sgp pallas --dry-run
python build_ccn_kappa.py && python build_composition.py --ccn-tier   # with the CCN tier
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sites import SITE_COORDS, SPECIES  # noqa: E402

warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", message=".*decode.*")

OBS = os.path.join(HERE, "data", "obs_comp")
CAMS_REGIONS = {  # same boxes as cams_download.REGIONS (not imported, to avoid needing cdsapi settings)
    "europe": ["puijo", "pallas", "kenttarova", "zeppelin", "ny-alesund", "hyytiala", "mace-head", "cabauw",
               "melpitz", "palaiseau", "bucharest", "granada", "schneefernerhaus", "norunda"],
    "sgp_bnf": ["sgp", "sgp_cloudnet", "bankhead"],
    "epcape": ["epcape_pier", "epcape_soledad"],
    "ena": ["ena", "ena_cloudnet"],
    "lasic": ["lasic", "lasic_cloudnet"],
}
ARM_GROUP = {"ena": "ena", "ena_cloudnet": "ena", "sgp": "sgp", "sgp_cloudnet": "sgp",
             "lasic": "lasic", "lasic_cloudnet": "lasic", "epcape_pier": "epcape",
             "epcape_soledad": "epcape", "bankhead": "bankhead"}

# molar masses (g mol-1)
M_SO4, M_NH4, M_NO3, M_CL, M_S, M_N = 96.06, 18.04, 62.00, 35.45, 32.06, 14.01
M_H2SO4, M_ABS, M_AS, M_AN, M_ACL = 98.08, 115.11, 132.14, 80.04, 53.49
SEASALT_PER_NA = 3.26
SEASALT_DRY_80RH = 4.3          # CAMS sea salt is at 80 % RH
MAC637 = 6.6                    # m2 g-1

# kappa and density (g cm-3) per PARSEC column as PARSEC's Koehler curve sees them: ideal solution,
# full dissociation (H2SO4, NH4HSO4, (NH4)2SO4: 3 ions; sea salt as NaCl: 2; organics: 1 ion at
# 200 g mol-1 with org_sol_frac = 1, i.e. kappa = n M_w rho / (rho_w M)); BC and dust insoluble.
# Densities from mo_parsec_m7par.f90. SVOC uses org_mol_mass / org_densities of the namelist (0.2, 1500).
KAPPA = np.array([1.014, 0.836, 0.724, 0.135, 0.0, 0.0, 1.335, 0.135])
RHO = np.array([1.841, 1.78, 1.77, 1.50, 2.00, 2.65, 2.165, 1.50])

ARM_ACSM_VARS = {"org": "total_organics", "so4": "sulfate", "nh4": "ammonium", "no3": "nitrate", "chl": "chloride"}


# ----------------------------------------------------------------------------- helpers
def std_atm(alt_m):
    """Standard-atmosphere pressure (hPa) and temperature (K) at altitude."""
    T = 288.15 - 0.0065 * alt_m
    return 1013.25 * (T / 288.15) ** 5.25588, T


def hour_index(hours):
    return pd.DatetimeIndex(hours)


def to_hours(start, end, df, hours, tol="60min"):
    """Values of df (rows = samples with [start, end)) for each hour in `hours`.

    Sub-hourly samples are averaged by the hour of their midpoint (an empty hour takes the nearest
    hourly mean within `tol`); samples of an hour or longer are looked up at the hour's midpoint."""
    out = pd.DataFrame(index=hours, columns=df.columns, dtype=float)
    if len(df) == 0:
        return out
    start, end = pd.DatetimeIndex(start), pd.DatetimeIndex(end)
    dur = (end - start).median()
    if pd.isna(dur) or dur < pd.Timedelta("55min"):
        mid = start + (end - start) / 2 if not pd.isna(dur) else start
        g = df.set_index(pd.DatetimeIndex(mid).floor("h")).groupby(level=0).mean()
        g = g[~g.index.duplicated()].sort_index()
        return g.reindex(hours, method="nearest", tolerance=pd.Timedelta(tol))
    order = np.argsort(start.values)
    s, e, v = start.values[order], end.values[order], df.to_numpy(float)[order]
    mids = (hours + pd.Timedelta("30min")).values
    k = np.searchsorted(s, mids, side="right") - 1
    ok = (k >= 0) & (mids < e[np.clip(k, 0, None)])
    arr = np.full((len(hours), df.shape[1]), np.nan)
    arr[ok] = v[k[ok]]
    return pd.DataFrame(arr, index=hours, columns=df.columns)


# ----------------------------------------------------------------------------- ARM
def arm_bad(ds, var):
    """Mask of samples flagged Bad in ARM's qc_<var> (bit assessments), or missing values."""
    x = ds[var].values.astype(float)
    bad = ~np.isfinite(x) | (x <= -9000)
    q = ds.get(f"qc_{var}")
    if q is not None:
        qv = q.values.astype(np.int64)
        for k in range(1, 33):
            if str(q.attrs.get(f"bit_{k}_assessment", "")).lower() == "bad":
                bad |= (qv & (1 << (k - 1))) > 0
    return bad


def arm_files(group, stream):
    return sorted(glob.glob(os.path.join(OBS, "arm", group, stream, "*.nc")) +
                  glob.glob(os.path.join(OBS, "arm", group, stream, "*.cdf")))


def arm_bounds(ds):
    t = pd.DatetimeIndex(ds["time"].values)
    if "time_bounds" in ds:
        b = ds["time_bounds"].values
        return pd.DatetimeIndex(b[:, 0]), pd.DatetimeIndex(b[:, 1])
    step = pd.Series(t).diff().median() if len(t) > 1 else pd.Timedelta("1min")
    return t, t + step


def read_arm_acsm(group, hours):
    """Hourly ACSM species (ug m-3), b2 hours first, gaps filled from b1 streams."""
    from obs_composition_download import ARM_GROUPS
    res = None
    for stream in ARM_GROUPS[group][1].get("acsm", []):
        rows = []
        for f in arm_files(group, stream):
            try:
                with xr.open_dataset(f) as ds:
                    if not all(v in ds for v in ARM_ACSM_VARS.values()):
                        continue
                    s, e = arm_bounds(ds)
                    d = {}
                    for k, v in ARM_ACSM_VARS.items():
                        x = ds[v].values.astype(float)
                        x[arm_bad(ds, v)] = np.nan
                        d[k] = x
                    rows.append(pd.DataFrame(dict(start=s, end=e, **d)))
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        if not rows:
            continue
        df = pd.concat(rows).dropna(subset=["so4", "org"])
        h = to_hours(df.start, df.end, df[list(ARM_ACSM_VARS)], hours)
        print(f"    {stream}: {len(df)} samples, {h['so4'].notna().sum()} hours")
        res = h if res is None else res.combine_first(h)
    return res


def read_arm_bc(group, hours):
    """Hourly eBC (ug m-3) from the PSAP red channel (Weiss-corrected), PM1 impactor where it cycles."""
    from obs_composition_download import ARM_GROUPS
    res = None
    for stream in ARM_GROUPS[group][1].get("black_carbon", []):
        if "psap" not in stream:
            continue  # SP2 not handled yet
        rows = []
        for f in arm_files(group, stream):
            try:
                with xr.open_dataset(f) as ds:
                    var = next((v for v in ("Ba_R_Weiss", "Ba_R_Bond", "Ba_R_raw") if v in ds), None)
                    if var is None:
                        continue
                    lam = float(ds[var].attrs.get("wavelength", 660.0) or 660.0)
                    x = ds[var].values.astype(float)
                    x[arm_bad(ds, var)] = np.nan
                    if "impactor_state" in ds:
                        imp = ds["impactor_state"].values
                        if np.any(imp == 1) and np.any(imp == 10):
                            x[imp != 1] = np.nan
                    s, e = arm_bounds(ds)
                    rows.append(pd.DataFrame(dict(start=s, end=e, bc=x / (MAC637 * 637.0 / lam))))
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        if not rows:
            continue
        df = pd.concat(rows).dropna(subset=["bc"])
        h = to_hours(df.start, df.end, df[["bc"]], hours)
        print(f"    {stream}: {h['bc'].notna().sum()} hours of eBC")
        res = h if res is None else res.combine_first(h)
    return res


# ----------------------------------------------------------------------------- EBAS
def ebas_valid(ds, var):
    """True where EBAS flags mark the value valid (approximate: flags 0-399, 499, 559, 640, 750-799)."""
    q = ds.get(f"{var}_qc")
    x = ds[var]
    if q is None:
        return xr.ones_like(x, dtype=bool)
    fdim = [d for d in q.dims if d.endswith("_qc_flags")]
    f = q.fillna(0).astype(int)
    ok = (f < 400) | (f == 499) | (f == 559) | (f == 640) | ((f >= 750) & (f < 800))
    return ok.all(dim=fdim) if fdim else ok


def ebas_series(ds, var, scale=1.0):
    x = ds[var].where(ebas_valid(ds, var)) * scale
    return x.values.astype(float)


def ebas_bounds(ds):
    b = ds["time_bnds"].values
    return pd.DatetimeIndex(b[:, 0]), pd.DatetimeIndex(b[:, 1])


def pick(ds, *names):
    return next((n for n in names if n in ds.data_vars), None)


def ebas_files(site, category):
    p = os.path.join(OBS, "manifest.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    m = pd.read_csv(p)
    m = m[(m.source == "EBAS") & (m.site == site) & (m.category == category)]
    return [os.path.join(HERE, x) for x in m.path.unique() if os.path.exists(os.path.join(HERE, x))]


def read_ebas_acsm(site, hours):
    names = {"org": ("organic_mass_amean", "organic_mass"),
             "so4": ("sulphate_total_ug_per_m3_amean", "sulphate_total_ug_per_m3", "sulphate_total"),
             "nh4": ("ammonium_ug_per_m3_amean", "ammonium_ug_per_m3", "ammonium"),
             "no3": ("nitrate_ug_per_m3_amean", "nitrate_ug_per_m3", "nitrate"),
             "chl": ("chloride_amean", "chloride")}
    res = None
    for f in ebas_files(site, "acsm_ams"):
        with xr.open_dataset(f) as ds:
            s, e = ebas_bounds(ds)
            d = {}
            for k, cands in names.items():
                v = pick(ds, *cands)
                d[k] = ebas_series(ds, v) if v else np.full(len(s), np.nan)
            df = pd.DataFrame(d).dropna(subset=["so4", "org"])
            h = to_hours(s[df.index], e[df.index], df, hours)
        print(f"    {os.path.basename(f)[:60]}: {h['so4'].notna().sum()} hours")
        res = h if res is None else res.combine_first(h)
    return res


def _ebas_var(ds, prefix, exclude=("_qc", "_ebasmetadata", "uncertainty", "perc", "prec", "stddev")):
    """First data variable starting with `prefix` (amean preferred) and its factor to ug m-3 of the ion
    (from 'ug S/m3' or 'ug N/m3' units or names)."""
    c = [v for v in ds.data_vars if v.startswith(prefix) and not any(x in v for x in exclude)]
    if not c:
        return None, 1.0
    v = sorted(c, key=lambda v: ("ug S" in v or "ug N" in v, not v.endswith("_amean")))[0]
    u = (str(ds[v].attrs.get("units", "")) + " " + v).replace("_", " ")
    f = M_SO4 / M_S if "ug S" in u else (M_NH4 / M_N if "ug N" in u else 1.0)
    return v, f


def read_ebas_filter(site, hours):
    """Daily filter ions (ug m-3): nss sulfate, ammonium, sodium. Variable names and S/N units are
    matched loosely, since older EBAS files name them differently."""
    res = None
    for f in ebas_files(site, "filter_ions"):
        with xr.open_dataset(f) as ds:
            s, e = ebas_bounds(ds)
            n = len(s)
            col = lambda pre: (lambda v, k: ebas_series(ds, v, k) if v else np.full(n, np.nan))(*_ebas_var(ds, pre))  # noqa: E731
            na = col("sodium")
            so4 = col("sulphate_corrected")
            tot = col("sulphate_total")
            so4 = np.where(np.isfinite(so4), so4, tot - 0.252 * np.nan_to_num(na))
            nh4 = col("ammonium")
            df = pd.DataFrame(dict(so4=so4, nh4=nh4, na=na)).dropna(subset=["so4"])
            h = to_hours(s[df.index], e[df.index], df, hours)
        print(f"    {os.path.basename(f)[:60]}: {h['so4'].notna().sum()} hours")
        res = h if res is None else res.combine_first(h)
    return res


def read_ebas_bc(site, hours):
    """Hourly eBC (ug m-3): equivalent_black_carbon if reported, else absorption nearest 637 nm / MAC."""
    res = None
    for f in ebas_files(site, "black_carbon"):
        try:
            with xr.open_dataset(f) as ds:
                s, e = ebas_bounds(ds)
                v = pick(ds, "equivalent_black_carbon_amean", "equivalent_black_carbon",
                         "elemental_carbon")
                if v:
                    x = ds[v].where(ebas_valid(ds, v))
                    scale = 1.0
                else:
                    v = pick(ds, "aerosol_absorption_coefficient_amean", "aerosol_absorption_coefficient")
                    if v is None:
                        continue
                    x = ds[v].where(ebas_valid(ds, v))
                    scale = None
                if "Wavelength" in x.dims:
                    lams = ds["Wavelength"].values.astype(float)
                    i = int(np.argmin(np.abs(lams - 637.0)))
                    x = x.isel(Wavelength=i)
                    lam = lams[i]
                else:
                    lam = 637.0
                for d in [d for d in x.dims if d != "time"]:
                    x = x.isel({d: 0})
                x = x.values.astype(float)
                if scale is None:
                    x = x / (MAC637 * 637.0 / lam)
                if str(ds[v].attrs.get("units", "")).startswith("ng"):
                    x = x / 1000.0
                df = pd.DataFrame(dict(bc=x)).dropna()
                h = to_hours(s[df.index], e[df.index], df, hours)
            print(f"    {os.path.basename(f)[:60]}: {h['bc'].notna().sum()} hours of eBC")
            res = h if res is None else res.combine_first(h)
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)[:60]}: {str(ex)[:80]}")
    return res


# ----------------------------------------------------------------------------- CAMS
CAMS_MAP = {"so4": ["aermr11"], "om": ["aermr07", "aermr08"], "bc": ["aermr09", "aermr10"],
            "ss1": ["aermr01"], "ss2": ["aermr02"], "ss3": ["aermr03"], "du1": ["aermr04"],
            "du2": ["aermr05"], "du3": ["aermr06"]}


def _cams_point(path, lat, lon, alt):
    with xr.open_dataset(path) as ds:
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        lonv = lon % 360 if float(ds["longitude"].max()) > 180 else lon
        p = ds.sel(latitude=lat, longitude=lonv, method="nearest")
        ps, _ = std_atm(alt)
        lev = float(p["pressure_level"].sel(pressure_level=min(ps, 1000.0), method="nearest"))
        p = p.sel(pressure_level=lev)
        z = 44330.8 * (1 - (lev / 1013.25) ** 0.190263)
        rho = lev * 100.0 / (287.05 * (288.15 - 0.0065 * z))          # kg m-3
        cols = {}
        for k, vs in CAMS_MAP.items():
            have = [v for v in vs if v in p]
            if have:
                a = sum(p[v] for v in have)
                for d in [d for d in a.dims if d != tname]:
                    a = a.isel({d: 0})
                cols[k] = a.values.astype(float) * rho * 1e9          # ug m-3
        return pd.DataFrame(cols, index=pd.DatetimeIndex(p[tname].values))


def load_cams(site):
    region = next((r for r, m in CAMS_REGIONS.items() if site in m), None)
    lat, lon, alt = SITE_COORDS[site]
    out = {}
    for mode, pat in (("3h", f"data/cams/{region}/eac4_{region}_*.nc"),
                      ("mon", f"data/cams_monthly/{region}/eac4m_{region}_*.nc")):
        parts = []
        for f in sorted(glob.glob(os.path.join(HERE, pat))):
            try:
                parts.append(_cams_point(f, lat, lon, alt))
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        out[mode] = pd.concat(parts).sort_index() if parts else None
    return out


def cams_for_hours(cams, hours):
    """CAMS (ug m-3) per hour and its source tag: 3-hourly within 2 h > monthly mean > calendar-month mean."""
    cols = list(CAMS_MAP)
    res = pd.DataFrame(np.nan, index=hours, columns=cols)
    tag = pd.Series("", index=hours, dtype=object)
    if cams.get("3h") is not None:
        c = cams["3h"]
        c = c[~c.index.duplicated()]
        r = c.reindex(hours, method="nearest", tolerance=pd.Timedelta("2h"))
        m = r[cols].notna().all(axis=1)
        res[m] = r[m]
        tag[m] = "cams3h"
    mon = cams.get("mon")
    if mon is not None:
        mon = mon[~mon.index.duplicated()]
        per = mon.index.to_period("M")
        need = res.isna().any(axis=1)
        hp = hours.to_period("M")
        lut = dict(zip(per, range(len(per))))
        idx = np.array([lut.get(p, -1) for p in hp])
        m = need.values & (idx >= 0)
        res.iloc[np.where(m)[0]] = mon.iloc[idx[m]][cols].values
        tag[m] = "camsmon"
        clim = mon.groupby(mon.index.month)[cols].mean()
        need = res.isna().any(axis=1).values
        mm = hours.month.values
        m = need & np.isin(mm, clim.index)
        res.iloc[np.where(m)[0]] = clim.loc[mm[m]].values
        tag[m] = "camsclim"
    return res, tag


# ----------------------------------------------------------------------------- composition
def sulfate_split(so4, nh4):
    """Masses of h2so4, nh4hso4, (nh4)2so4 from sulfate and ammonium (ug m-3); NaN ammonium = neutral."""
    n_s = so4 / M_SO4
    R = np.where(np.isfinite(nh4), np.clip(nh4 / M_NH4 / np.where(n_s > 0, n_s, np.nan), 0, 2), 2.0)
    R = np.where(np.isfinite(R), R, 2.0)
    as_ = np.where(R >= 1, (R - 1), 0) * n_s
    abs_ = np.where(R >= 1, 2 - R, R) * n_s
    h2 = np.where(R < 1, 1 - R, 0) * n_s
    return h2 * M_H2SO4, abs_ * M_ABS, as_ * M_AS


def ccn_kappa_for_hours(site, hours, target, window=0.15):
    """kappa_ccn per hour at the valid setpoint nearest `target` (within `window` %), from build_ccn_kappa.py."""
    p = os.path.join(HERE, "results", f"{site}_ccn_kappa.nc")
    out = pd.Series(np.nan, index=hours)
    if not os.path.exists(p):
        print(f"   --ccn-tier: no {os.path.relpath(p, HERE)} (run build_ccn_kappa.py first)")
        return out
    with xr.open_dataset(p) as c:
        ss = c["ss"].values.astype(float)
        k = c["kappa_ccn"].where(c["kappa_ok"]).to_pandas().reindex(hours)
    for j in np.argsort(np.abs(ss - target)):
        if abs(ss[j] - target) <= window + 1e-9:
            out = out.fillna(k.iloc[:, j])
    return out


def apply_ccn_kappa(mass, kc):
    """Re-split the (nh4)2so4 + organics volume of each row so the bulk kappa equals kc; BC, dust and
    sea salt are kept. Returns new masses and a mask of rows where the split hit 0 or 1."""
    m = mass.copy()
    v = m / RHO
    v_ao = v[:, 2] + v[:, 3]
    k_rest = (v[:, [0, 1, 4, 5, 6]] * KAPPA[[0, 1, 4, 5, 6]]).sum(axis=1)
    v_rest = v[:, [0, 1, 4, 5, 6]].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = ((kc * (v_rest + v_ao) - k_rest) / v_ao - KAPPA[3]) / (KAPPA[2] - KAPPA[3])
    clipped = np.isfinite(x) & ((x < 0) | (x > 1))
    x = np.clip(x, 0, 1)
    ok = np.isfinite(x) & (v_ao > 0)
    m[ok, 2] = x[ok] * v_ao[ok] * RHO[2]
    m[ok, 3] = (1 - x[ok]) * v_ao[ok] * RHO[3]
    return m, ok, clipped & ok


def build_site(site, dry_run=False, ccn_tier=False, ccn_ss=0.3, cams_only=False, suffix=""):
    path = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(path):
        return None
    out_path = os.path.join(HERE, "inputs", f"{site}_parcel_inputs{suffix}.nc")
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    hours = hour_index(ds["hour"].values)
    nh = len(hours)
    print(f"\n{site}: {nh} hours")

    acsm = bc = filt = None
    if site in ARM_GROUP:
        g = ARM_GROUP[site]
        acsm = read_arm_acsm(g, hours)
        bc = read_arm_bc(g, hours)
    e_acsm, e_bc, filt = read_ebas_acsm(site, hours), read_ebas_bc(site, hours), read_ebas_filter(site, hours)
    acsm = e_acsm if acsm is None else (acsm if e_acsm is None else acsm.combine_first(e_acsm))
    bc = e_bc if bc is None else (bc if e_bc is None else bc.combine_first(e_bc))

    cams, ctag = cams_for_hours(load_cams(site), hours)
    nan = pd.Series(np.nan, index=hours)
    A = acsm if acsm is not None else pd.DataFrame({k: nan for k in ARM_ACSM_VARS})
    F = filt if filt is not None else pd.DataFrame({k: nan for k in ("so4", "nh4", "na")})
    B = bc["bc"] if bc is not None else nan

    use_a = (A["so4"].notna() & A["org"].notna()).values
    use_f = ~use_a & F["so4"].notna().values
    have_bc = B.notna().values
    if cams_only:                       # uniform composition: CAMS for every site and hour, measurements ignored
        use_a[:] = False
        use_f[:] = False
        have_bc[:] = False
    have_c = cams.notna().all(axis=1).values

    so4 = np.where(use_a, A["so4"].clip(lower=0), np.where(use_f, F["so4"].clip(lower=0), cams["so4"]))
    nh4 = np.where(use_a, A["nh4"].clip(lower=0), np.where(use_f, F["nh4"].clip(lower=0), np.nan))
    no3 = np.where(use_a, A["no3"].clip(lower=0).fillna(0), 0.0)
    chl = np.where(use_a, A["chl"].clip(lower=0).fillna(0), 0.0)
    om_ratio = cams["om"] / cams["so4"].where(cams["so4"] > 0)
    org = np.where(use_a, A["org"].clip(lower=0), np.where(use_f, om_ratio * so4, cams["om"]))
    bcv = np.where(have_bc, B.clip(lower=0), cams["bc"])
    ss_frac = cams["ss1"] / (cams["ss1"] + cams["ss2"] + cams["ss3"]).where(lambda x: x > 0)
    ss_filter = SEASALT_PER_NA * F["na"].clip(lower=0) * ss_frac
    ss = np.where(use_f & ss_filter.notna().values, ss_filter, cams["ss1"] / SEASALT_DRY_80RH)
    dust = cams["du1"].values

    h2, abs_, as_ = sulfate_split(np.asarray(so4, float), np.asarray(nh4, float))
    as_ = as_ + no3 * M_AN / M_NO3 + chl * M_ACL / M_CL
    mass = np.column_stack([h2, abs_, as_, org, bcv, dust, ss, np.zeros(nh)]).astype(float)
    mass = np.where(np.isfinite(mass), np.clip(mass, 0, None), np.nan)
    # missing CAMS fills (no CAMS file at all): dust / sea salt / BC count as 0 rather than losing the hour
    mass[:, 4:7] = np.nan_to_num(mass[:, 4:7])
    tot = mass.sum(axis=1)
    good = np.isfinite(tot) & (tot > 0)

    tier = np.where(use_a, "acsm", np.where(use_f, "filter", np.where(have_bc, "bc", "cams")))
    if ccn_tier:
        kc = ccn_kappa_for_hours(site, hours, ccn_ss).values
        cand = np.isin(tier, ["bc", "cams"]) & np.isfinite(kc) & good
        m2, ok2, clip2 = apply_ccn_kappa(mass[cand], kc[cand])
        idx = np.where(cand)[0][ok2]
        mass[idx] = m2[ok2]
        tier = tier.astype(object)
        tier[idx] = "ccn"
        tot = mass.sum(axis=1)
        print(f"   --ccn-tier: {len(idx)} hours re-split to match kappa_ccn near {ccn_ss} % "
              f"({int(clip2.sum())} at the (nh4)2so4-only or organics-only limit)")
    bctag = np.where(have_bc, "+bc_obs", "+bc_cams")
    src = np.array([t + (b if t in ("acsm", "filter", "ccn") else "") + ("+" + c if c else "+nocams")
                    for t, b, c in zip(tier, bctag, ctag.values)], dtype=object)
    src[~good] = "regime_placeholder"

    mf = ds["massfrac"].values.copy() if "massfrac_placeholder" not in ds else ds["massfrac_placeholder"].values.copy()
    new = mf.copy()
    new[good] = mass[good] / tot[good, None]
    vol = np.where(good[:, None], new / RHO, np.nan)
    kappa = np.nansum(vol * KAPPA, axis=1) / np.nansum(vol, axis=1)

    s = pd.Series(src)
    summ = s.str.split("+").str[0].value_counts().to_dict()
    print(f"   tiers: {summ}")
    print(f"   cams:  {pd.Series([x.split('+')[-1] for x in src]).value_counts().to_dict()}")
    print(f"   median mass fractions: " + ", ".join(f"{k} {v:.2f}" for k, v in zip(SPECIES, np.nanmedian(new, axis=0))))
    print(f"   bulk kappa median {np.nanmedian(kappa):.2f} (10-90 %: {np.nanpercentile(kappa, 10):.2f}-{np.nanpercentile(kappa, 90):.2f})")
    if dry_run:
        return s

    if "massfrac_placeholder" not in ds:
        ds["massfrac_placeholder"] = ds["massfrac"].copy()
    ds["massfrac"] = (("hour", "species"), new, {"units": "1", "long_name": "dry mass fractions, PARSEC order"})
    ds["comp_source"] = ("hour", src)
    for j, k in enumerate(["m_h2so4", "m_nh4hso4", "m_nh42so4", "m_org", "m_bc", "m_dust", "m_seasalt"]):
        ds[k] = ("hour", mass[:, j], {"units": "ug m-3", "long_name": f"dry mass, {SPECIES[j]} column"})
    ds["kappa_bulk"] = ("hour", kappa, {"units": "1", "long_name": "volume-weighted bulk kappa (diagnostic)"})
    ds.attrs["composition"] = "build_composition.py: " + ", ".join(f"{k} {v}" for k, v in summ.items())
    tmp = out_path + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, out_path)
    print(f"   -> {os.path.relpath(out_path, HERE)}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--dry-run", action="store_true", help="report the tiers, do not write")
    ap.add_argument("--ccn-tier", action="store_true",
                    help="hours that would get only BC + CAMS or CAMS composition: split (nh4)2so4 vs organics "
                         "to match kappa_ccn from results/<site>_ccn_kappa.nc (build_ccn_kappa.py)")
    ap.add_argument("--cams-only", action="store_true",
                    help="uniform composition: CAMS for every site and hour (no ACSM, filter or eBC)")
    ap.add_argument("--suffix", default="",
                    help="write inputs/<site>_parcel_inputs<suffix>.nc instead of overwriting the inputs file, "
                         "e.g. --cams-only --suffix _cams (run parsec_run.py --inputs on it with another OUTDIR)")
    ap.add_argument("--ccn-ss", type=float, default=0.3, help="target setpoint for --ccn-tier, %% (default 0.3)")
    a = ap.parse_args()
    sites = a.sites or sorted(os.path.basename(p).replace("_parcel_inputs.nc", "")
                              for p in glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))
    rows = []
    for s in sites:
        r = build_site(s, a.dry_run, a.ccn_tier, a.ccn_ss, a.cams_only, a.suffix)
        if r is not None:
            for t, n in r.str.split("+").str[0].value_counts().items():
                rows.append(dict(site=s, tier=t, hours=n))
    if rows:
        t = pd.DataFrame(rows).pivot_table(index="site", columns="tier", values="hours", fill_value=0)
        print("\n", t.astype(int).to_string())


if __name__ == "__main__":
    main()
