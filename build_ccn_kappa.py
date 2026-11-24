#!/usr/bin/env python
"""Hygroscopicity from the CCN counters: kappa per cloud hour and supersaturation, plus a CCN closure.

For each hour in inputs/<site>_parcel_inputs.nc and each supersaturation step of the CCN counter:

  D_c        critical dry diameter: the size above which the hourly spectrum (dNdlogDp) holds as many
             particles as the counter measured
  kappa_ccn  4 A^3 / (27 D_c^3 ln^2 S), A = 4 sigma M_w / (R T rho_w) (sigma 0.072 N m-1, T 298.15 K)
  ccn_pred   CCN predicted from the spectrum and the composition's bulk kappa (inputs kappa_bulk,
             from build_composition.py), for the closure ratio ccn_pred / ccn_obs

Written to results/<site>_ccn_kappa.nc with dims (hour, ss), ss being the counter's supersaturation
level (setpoints, or calibrated S where no setpoint is stored, grouped where they differ by < 10 %). S is the counter's calibrated ("calculated") supersaturation where reported, else the
setpoint. D_c is flagged unreliable (kappa_ok = False) when the counter count exceeds the spectrum's
total (no D_c), D_c lies in the lowest two bins of the spectrum, or kappa falls outside 0.02-1.5.

build_composition.py --ccn-tier uses kappa_ccn at the setpoint nearest --ccn-ss to set the
inorganic/organic split for hours that would otherwise get only BC + CAMS or CAMS composition.

Sources: ARM ccn1colavg / ccn2colaavg streams (data/obs_comp/arm/<group>/), EBAS CCN files from the
manifest. Variable names are matched loosely (N_CCN*, cloud_condensation_nuclei*, supersaturation*);
the script prints what it used per file type.

Examples
--------
python build_ccn_kappa.py                 # all sites with an inputs file and CCN data
python build_ccn_kappa.py --sites hyytiala sgp
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_composition import (ARM_GROUP, OBS, arm_bad, arm_bounds, arm_files,  # noqa: E402
                               ebas_bounds, ebas_valid)

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

SIGMA, MW, RHOW, RGAS, T_CCN = 0.072, 0.018015, 997.0, 8.314, 298.15
A_KELVIN = 4 * SIGMA * MW / (RGAS * T_CCN * RHOW)          # m (~2.1 nm)
KAPPA_RANGE = (0.02, 1.5)
_seen = set()


def note(key, msg):
    if key not in _seen:
        _seen.add(key)
        print(msg)


def kappa_from_dc(dc_nm, s_pct):
    d = np.asarray(dc_nm, float) * 1e-9
    ln = np.log1p(np.asarray(s_pct, float) / 100.0)
    return 4 * A_KELVIN ** 3 / (27 * d ** 3 * ln ** 2)


def dc_from_kappa(kappa, s_pct):
    ln = np.log1p(np.asarray(s_pct, float) / 100.0)
    return (4 * A_KELVIN ** 3 / (27 * np.asarray(kappa, float) * ln ** 2)) ** (1 / 3) * 1e9


def cum_above(spec, d_nm):
    """N(>D) at the bin edges (cm-3) for spectra (hour, bin) of dN/dlogDp on log-spaced centres d_nm."""
    ld = np.log10(d_nm)
    edges = np.r_[ld[0] - (ld[1] - ld[0]) / 2, (ld[1:] + ld[:-1]) / 2, ld[-1] + (ld[-1] - ld[-2]) / 2]
    dN = np.nan_to_num(spec) * np.diff(edges)
    above = np.cumsum(dN[:, ::-1], axis=1)[:, ::-1]                      # N above lower edge of bin i
    return edges, np.concatenate([above, np.zeros((spec.shape[0], 1))], axis=1)


def invert_dc(edges, above, n_ccn):
    """D_c (nm) with N(>D_c) = n_ccn, log-linear between edges; NaN where n_ccn exceeds the total."""
    dc = np.full(n_ccn.shape, np.nan)
    for i in range(above.shape[0]):
        a = above[i][::-1]                                                # increasing
        e = edges[::-1]
        for j in range(n_ccn.shape[1]):
            n = n_ccn[i, j]
            if not np.isfinite(n) or n <= 0 or n > a[-1]:
                continue
            k = np.searchsorted(a, n)
            if k == 0:
                continue
            f = (n - a[k - 1]) / (a[k] - a[k - 1]) if a[k] > a[k - 1] else 0.0
            dc[i, j] = 10 ** (e[k - 1] + f * (e[k] - e[k - 1]))
    return dc


def n_above(edges, above, d_nm):
    out = np.full(d_nm.shape, np.nan)
    for i in range(above.shape[0]):
        ok = np.isfinite(d_nm[i])
        out[i, ok] = np.interp(np.log10(d_nm[i, ok]), edges, above[i])
    return out


# ----------------------------------------------------------------------------- readers
def _samples_arm(ds):
    """(start, end, ss_set, ss_eff, n) samples from an ARM CCN file (one or two columns)."""
    s, e = arm_bounds(ds)
    out = []
    cands = [v for v in ds.data_vars
             if not v.startswith("qc_") and re.match(r"(?i)^n_ccn", v) and ds[v].dims == ("time",)]
    if "N_CCN" in cands:                 # LASIC also has N_CCN_reduced etc.: use the standard one only
        cands = ["N_CCN"]
    for v in cands:
        suf = v[5:]
        set_v = next((c for c in (f"supersaturation_setpoint{suf}", "supersaturation_setpoint") if c in ds), None)
        eff_v = next((c for c in (f"supersaturation_calculated{suf}", "supersaturation_calculated") if c in ds), None)
        if set_v is None and eff_v is None:
            continue
        n = ds[v].values.astype(float)
        n[arm_bad(ds, v)] = np.nan
        sset = ds[set_v].values.astype(float) if set_v else ds[eff_v].values.astype(float)
        seff = ds[eff_v].values.astype(float) if eff_v else sset
        seff = np.where((seff > 0) & (seff < 5), seff, sset)
        note(("arm", v, set_v, eff_v), f"    ARM: {v} with {set_v} / {eff_v}")
        out.append(pd.DataFrame(dict(start=s, end=e, ss_set=sset, ss_eff=seff, n=n)))
    return out


def read_arm(site):
    from obs_composition_download import ARM_GROUPS
    g = ARM_GROUP.get(site)
    if g is None:
        return []
    rows = []
    for stream in ARM_GROUPS[g][1].get("ccn", []):
        files = arm_files(g, stream)
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    rows += _samples_arm(ds)
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        if files:
            print(f"    {stream}: {len(files)} files")
    return rows


def ccn_files_for(site):
    """EBAS CCN files of every station this site uses (Zeppelin / Ny-Alesund and Pallas / Kenttarova
    share stations, but the manifest lists a CCN file only for the site whose hours it covered)."""
    p = os.path.join(OBS, "manifest.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    m = pd.read_csv(p)
    m = m[m.source == "EBAS"]
    st = m.path.str.split("/").str[-2]
    mine = set(st[m.site == site])
    files = m.path[(m.category == "ccn") & st.isin(mine)].unique()
    return [os.path.join(HERE, x) for x in files if os.path.exists(os.path.join(HERE, x))]


def read_ebas(site):
    rows = []
    for f in ccn_files_for(site):
        try:
            with xr.open_dataset(f) as ds:
                s, e = ebas_bounds(ds)
                cands = [v for v in ds.data_vars if "cloud_condensation_nuclei" in v
                         and not v.endswith(("_qc", "_ebasmetadata")) and "uncertainty" not in v
                         and "perc" not in v and "prec" not in v and "stddev" not in v]
                v = next((c for c in cands if c.endswith("_amean")), cands[0] if cands else None)
                if v is None:
                    print(f"    no CCN variable in {os.path.basename(f)[:60]}")
                    continue
                x = ds[v].where(ebas_valid(ds, v))
                ssdim = next((d for d in x.dims if d != "time" and ("supersat" in d.lower() or d.upper() == "SS")), None)
                for d in [d for d in x.dims if d not in ("time", ssdim)]:
                    x = x.isel({d: 0})
                if ssdim is not None:
                    for k, ssv in enumerate(ds[ssdim].values.astype(float)):
                        rows.append(pd.DataFrame(dict(start=s, end=e, ss_set=ssv, ss_eff=ssv,
                                                      n=x.isel({ssdim: k}).values.astype(float))))
                    note(("ebas", v, ssdim), f"    EBAS: {v} over dimension {ssdim}")
                else:
                    sv = next((c for c in ds.data_vars if "supersaturation" in c.lower()
                               and not c.endswith(("_qc", "_ebasmetadata"))), None)
                    if sv is None:
                        print(f"    no supersaturation in {os.path.basename(f)[:60]}")
                        continue
                    ssv = ds[sv].values.astype(float).ravel()[: len(s)]
                    rows.append(pd.DataFrame(dict(start=s, end=e, ss_set=ssv, ss_eff=ssv, n=x.values.astype(float))))
                    note(("ebas", v, sv), f"    EBAS: {v} with {sv}")
            print(f"    {os.path.basename(f)[:70]}")
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)[:60]}: {str(ex)[:80]}")
    return rows


def ss_levels(x, gap=0.10):
    """Group supersaturations into levels: a new level starts where sorted values jump by > `gap`
    (relative). Returns each value's level label (median of the level, rounded to 0.01 %)."""
    u = np.unique(np.round(x, 4))
    lab = np.zeros(len(u), int)
    for i in range(1, len(u)):
        lab[i] = lab[i - 1] + (u[i] > u[i - 1] * (1 + gap))
    med = pd.Series(u).groupby(lab).median().round(2).values
    return med[lab[np.searchsorted(u, np.round(x, 4))]]


def hourly(rows, hours):
    """Hourly mean CCN and effective S per supersaturation level: DataFrames (hour x ss).

    Samples shorter than ~1 h go to the hour of their midpoint; longer ones (e.g. 4-h EBAS averages)
    to every hour whose midpoint they cover. Levels are clusters of the setpoint (or of the calibrated
    S where no setpoint is reported), so day-to-day drift in the calibrated S stays in one level."""
    if not rows:
        return None, None
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df.n) & (df.n >= 0) & np.isfinite(df.ss_set) & (df.ss_set > 0.02)]
    if df.empty:
        return None, None
    st, en = pd.DatetimeIndex(df.start), pd.DatetimeIndex(df.end)
    long_ = (en - st) >= pd.Timedelta("55min")
    df = df.assign(hour=(st + (en - st) / 2).floor("h"))
    if long_.any():
        lo = df[long_]
        lo = lo[(pd.DatetimeIndex(lo.end) > hours.min()) & (pd.DatetimeIndex(lo.start) <= hours.max() + pd.Timedelta("1h"))]
        hs = [pd.date_range((a - pd.Timedelta("30min")).ceil("h"), b - pd.Timedelta("30min") - pd.Timedelta("1ns"), freq="h")
              for a, b in zip(pd.DatetimeIndex(lo.start), pd.DatetimeIndex(lo.end))]
        lo = lo.assign(hour=hs).explode("hour").dropna(subset=["hour"])
        df = pd.concat([df[~long_], lo], ignore_index=True)
        df["hour"] = pd.DatetimeIndex(df.hour)
    df = df[df.hour.isin(hours)]
    if df.empty:
        return None, None
    df = df.assign(ss=ss_levels(df.ss_set.values))
    if df.empty:
        return None, None
    g = df.groupby(["hour", "ss"])
    n = g.n.mean().unstack().reindex(hours)
    s = g.ss_eff.mean().unstack().reindex(hours)
    return n, s


# ----------------------------------------------------------------------------- per site
def build_site(site):
    path = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(path):
        return None
    rows = read_arm(site) + read_ebas(site)
    if not rows:
        return None
    with xr.open_dataset(path) as ds:
        hours = pd.DatetimeIndex(ds["hour"].values)
        spec = ds["dNdlogDp"].values.astype(float)
        d_nm = ds["diameter"].values.astype(float)
        kb = ds["kappa_bulk"].values.astype(float) if "kappa_bulk" in ds else np.full(len(hours), np.nan)
        src = ds["comp_source"].values.astype(str)
    print(f"\n{site}: {len(hours)} hours")
    n, s = hourly(rows, hours)
    if n is None:
        print("   no CCN samples in the cloud hours")
        return None
    ss = n.columns.values
    N, S = n.to_numpy(float), s.to_numpy(float)
    edges, above = cum_above(spec, d_nm)
    dc = invert_dc(edges, above, N)
    kap = kappa_from_dc(dc, S)
    ok = np.isfinite(kap) & (kap >= KAPPA_RANGE[0]) & (kap <= KAPPA_RANGE[1]) & (dc > 10 ** edges[2])
    dcp = dc_from_kappa(np.repeat(kb[:, None], len(ss), axis=1), S)
    pred = n_above(edges, above, dcp)

    out = xr.Dataset(
        {"ccn_obs": (("hour", "ss"), N, {"units": "cm-3"}),
         "ss_eff": (("hour", "ss"), S, {"units": "%", "long_name": "effective supersaturation (hourly mean)"}),
         "dc": (("hour", "ss"), dc, {"units": "nm", "long_name": "critical dry diameter from the spectrum"}),
         "kappa_ccn": (("hour", "ss"), kap, {"units": "1"}),
         "kappa_ok": (("hour", "ss"), ok),
         "ccn_pred": (("hour", "ss"), pred, {"units": "cm-3",
                                             "long_name": "CCN from the spectrum and kappa_bulk"}),
         "kappa_bulk": ("hour", kb), "comp_source": ("hour", src.astype(object)),
         "n_total": ("hour", above[:, 0], {"units": "cm-3", "long_name": "spectrum total"})},
        coords={"hour": hours.values, "ss": ("ss", ss, {"units": "%", "long_name": "supersaturation level (setpoint cluster)"})},
        attrs={"T_kelvin": T_CCN, "sigma": SIGMA})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    op = os.path.join(HERE, "results", f"{site}_ccn_kappa.nc")
    out.to_netcdf(op)

    tier = pd.Series(src).str.split("+").str[0].values
    print(f"   {'SS %':>5s} {'hours':>6s} {'kappa ok':>8s} {'Dc nm':>6s} {'kappa_ccn':>9s} "
          f"{'kappa_bulk':>10s} {'pred/obs':>8s}")
    for j, x in enumerate(ss):
        h = np.isfinite(N[:, j])
        o = ok[:, j]
        r = pred[o, j] / N[o, j]
        print(f"   {x:5.2f} {h.sum():6d} {o.sum():8d} {np.nanmedian(dc[o, j]) if o.any() else np.nan:6.0f} "
              f"{np.nanmedian(kap[o, j]) if o.any() else np.nan:9.2f} {np.nanmedian(kb[o]) if o.any() else np.nan:10.2f} "
              f"{np.nanmedian(r) if o.any() else np.nan:8.2f}")
    anyok = ok.any(axis=1)
    print("   hours with a valid kappa, by composition tier: "
          + ", ".join(f"{t} {int(anyok[tier == t].sum())}/{int((tier == t).sum())}" for t in pd.unique(tier)))
    print(f"   -> {os.path.relpath(op, HERE)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    a = ap.parse_args()
    sites = a.sites or sorted(os.path.basename(p).replace("_parcel_inputs.nc", "")
                              for p in glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))
    for s in sites:
        build_site(s)


if __name__ == "__main__":
    main()
