#!/usr/bin/env python
"""Cloud-base temperature and pressure (parcel start T0, P0) from surface observations, with ERA5 as fallback.

For each hour of inputs/<site>_parcel_inputs.nc without a measured cloud-base value (NDROP built files,
Zeppelin station), in order:

  surface_dry_adiabat   surface T, RH, p (data/met/ from met_download.py) taken along a dry adiabat to the
                        measured cloud base, when cloud base is within --coupled m (300) of the surface LCL,
                        i.e. the sub-cloud layer is coupled to the surface
  station_met           in-cloud stations (Puijo, Pallas, Mt Soledad): the station's own T and p
  <model>_decoupled     model at cloud base when the cloud is decoupled from the surface
  <model>_cloud_base /  model where there is no surface met (T linear, ln p linear in height; site altitude
  <model>_station       + cloud base height, or the station). <model> is cloudnet_model (the model profiles
                        in the Cloudnet categorize files from updraft_download.py) where those exist, else
                        era5 (era5_download.py)
  surface_decoupled     surface dry adiabat for decoupled hours without ERA5

All candidates are stored (T_surface_cb, P_surface_cb, T_model, P_model, cbh_minus_lcl). The script prints
surface minus model, and at the NDROP sites both minus the measured cloud-base values, as checks.

Needs data/met/ (met_download.py) and/or data/era5/ (era5_download.py); either alone works.

Examples
--------
python build_thermo.py
python build_thermo.py --sites hyytiala norunda --dry-run
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from era5_download import cell_name, cell_of  # noqa: E402
from sites import SITE_COORDS, SITES  # noqa: E402

G0 = 9.80665


def load_era5(site):
    lat, lon, _ = SITE_COORDS[site]
    name = cell_name(cell_of(site))
    parts = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "era5", name, f"era5_{name}_*.nc"))):
        try:
            with xr.open_dataset(f) as ds:
                tn = "valid_time" if "valid_time" in ds.dims else "time"
                ln = "pressure_level" if "pressure_level" in ds.dims else "level"
                p = ds.sel(latitude=lat, longitude=lon, method="nearest")
                for d in [d for d in p["t"].dims if d not in (tn, ln)]:
                    p = p.isel({d: 0})
                T = p["t"].transpose(tn, ln).values.astype(float)
                Z = p["z"].transpose(tn, ln).values.astype(float) / G0
                lev = p[ln].values.astype(float)
                parts.append((pd.DatetimeIndex(p[tn].values), lev, T, Z))
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
    return parts


def interp_at(parts, hours, z_target):
    """ERA5 T (K) and p (Pa) at height z_target (m a.s.l.) for each hour (exact hourly match)."""
    T0 = np.full(len(hours), np.nan)
    P0 = np.full(len(hours), np.nan)
    for t, lev, T, Z in parts:
        idx = pd.Index(t).get_indexer(hours)
        for i in np.where(idx >= 0)[0]:
            if not np.isfinite(z_target[i]):
                continue
            z, tt = Z[idx[i]], T[idx[i]]
            o = np.argsort(z)
            z, tt, lp = z[o], tt[o], np.log(lev[o] * 100.0)
            if z_target[i] < z[0]:                       # below the lowest level: standard lapse, hypsometric
                dz = z_target[i] - z[0]
                T0[i] = tt[0] - 0.0065 * dz
                P0[i] = np.exp(lp[0]) * np.exp(-G0 * dz / (287.05 * 0.5 * (tt[0] + T0[i])))
            elif z_target[i] <= z[-1]:
                T0[i] = np.interp(z_target[i], z, tt)
                P0[i] = np.exp(np.interp(z_target[i], z, lp))
    return T0, P0


MEASURED = ("built_file_cloud_base", "zeppelin_station")       # never replaced
# surface met measured at another altitude than the site (m a.s.l.): Mt Soledad uses the pier MET
MET_ALT = {"epcmetM1.b1": 10.0}
G_CP = G0 / 1004.0                                              # dry-adiabatic lapse rate, K m-1


def _hourly(df, hours):
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).floor("h")
    return df.groupby(level=0).mean().reindex(hours)


def load_surface(site, hours):
    """Hourly surface T (K), RH (%), p (Pa) and the source name, or (None, None)."""
    from build_composition import arm_bad
    from met_download import ARM_MET
    from updraft_search import CLOUDNET_ALIASES
    met = os.path.join(HERE, "data", "met")
    for stream in ARM_MET.get(site, []):
        files = sorted(glob.glob(os.path.join(met, "arm", stream, "*.nc")) + glob.glob(os.path.join(met, "arm", stream, "*.cdf")))
        parts = []
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    d = {}
                    for k, v in (("T", "temp_mean"), ("RH", "rh_mean"), ("P", "atmos_pressure")):
                        x = ds[v].values.astype(float)
                        x[arm_bad(ds, v)] = np.nan
                        d[k] = x
                    df = pd.DataFrame(d, index=pd.DatetimeIndex(ds["time"].values))
                    u = str(ds["atmos_pressure"].attrs.get("units", "kPa")).lower()
                    df["P"] *= 1000.0 if "kpa" in u else (100.0 if "hpa" in u or "mb" in u else 1.0)
                    if str(ds["temp_mean"].attrs.get("units", "degC")).lower() in ("degc", "c", "celsius"):
                        df["T"] += 273.15
                    parts.append(df)
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        if parts:
            return _hourly(pd.concat(parts), hours), f"arm_{stream}"
    # (MET_ALT keys are the stream names; build_site reads the altitude from the returned source name)
    for cid in CLOUDNET_ALIASES.get(site, [site]):
        files = sorted(glob.glob(os.path.join(met, "cloudnet", cid, "*.nc")))
        parts = []
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    df = pd.DataFrame({"T": ds["air_temperature"].values.astype(float),
                                       "RH": ds["relative_humidity"].values.astype(float),
                                       "P": ds["air_pressure"].values.astype(float)},
                                      index=pd.DatetimeIndex(ds["time"].values))
                    if np.nanmax(df["RH"]) <= 1.5:
                        df["RH"] *= 100.0
                    if np.nanmedian(df["P"]) < 2000:
                        df["P"] *= 100.0
                    if np.nanmedian(df["T"]) < 150:
                        df["T"] += 273.15
                    parts.append(df)
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        if parts:
            return _hourly(pd.concat(parts), hours), f"cloudnet_{cid}_weather_station"
    f = os.path.join(met, "fmi", f"{site}.csv")
    if os.path.exists(f):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None) - pd.Timedelta("1h")   # FMI stamps hour end
        df = pd.DataFrame({"T": df["TA_PT1H_AVG"] + 273.15, "RH": df["RH_PT1H_AVG"], "P": df["PA_PT1H_AVG"] * 100.0})
        return _hourly(df, hours), "fmi_station"
    return None, None


def load_cloudnet_model(site, hours, z_target):
    """T (K) and p (Pa) at z_target (m a.s.l.) from the model profiles in the Cloudnet categorize files
    downloaded for the updrafts (nearest model hour within 1 h; T linear, ln p linear in height)."""
    from updraft_download import CLOUDNET_RADAR_SITES
    cid = CLOUDNET_RADAR_SITES.get(site)
    T = np.full(len(hours), np.nan)
    P = np.full(len(hours), np.nan)
    files = sorted(glob.glob(os.path.join(HERE, "data", "updraft", "cloudnet_categorize", cid or "-", "*.nc")))
    pos = {h: i for i, h in enumerate(hours)}
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                mt = pd.DatetimeIndex(ds["model_time"].values)
                zh = ds["model_height"].values.astype(float)
                tt = ds["temperature"].values.astype(float)
                pp = ds["pressure"].values.astype(float)
            for k, t in enumerate(mt):
                i = pos.get(t.round("h"))
                if i is None or not np.isfinite(z_target[i]) or abs((t - t.round("h")).total_seconds()) > 3600:
                    continue
                z = zh if zh.ndim == 1 else zh[k]
                o = np.argsort(z)
                ok = np.isfinite(tt[k][o]) & np.isfinite(pp[k][o]) & (pp[k][o] > 0)
                if ok.sum() < 3 or not (z[o][ok][0] <= z_target[i] <= z[o][ok][-1]):
                    continue
                T[i] = np.interp(z_target[i], z[o][ok], tt[k][o][ok])
                P[i] = np.exp(np.interp(z_target[i], z[o][ok], np.log(pp[k][o][ok])))
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
    return T, P


def surface_to_cloud_base(T, RH, P, cbh):
    """Dry adiabat from the surface to cloud base; also the LCL (Espy, 125 m per K of dew-point depression)."""
    a, b = 17.625, 243.04
    tc = T - 273.15
    g = np.log(np.clip(RH, 1, 100) / 100.0) + a * tc / (b + tc)
    td = b * g / (a - g)
    z_lcl = 125.0 * (tc - td)
    Tcb = T - G_CP * cbh
    Pcb = P * (Tcb / T) ** 3.5
    return Tcb, Pcb, z_lcl


def build_site(site, dry_run=False, coupled_m=300.0):
    path = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(path):
        return
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    hours = pd.DatetimeIndex(ds["hour"].values)
    kind = SITES[site]["kind"]
    alt = SITE_COORDS[site][2]
    cbh = ds["cloud_base_height"].values.astype(float) if "cloud_base_height" in ds else np.full(len(hours), np.nan)
    z_t = np.full(len(hours), alt) if kind == "insitu" else alt + cbh
    print(f"\n{site} ({kind}): {len(hours)} hours, station {alt:.0f} m"
          + ("" if kind == "insitu" else f", cloud base median {np.nanmedian(cbh):.0f} m above ground"))

    # model fallback: Cloudnet categorize model profiles first (no queue), ERA5 where those are missing
    T_c, P_c = load_cloudnet_model(site, hours, z_t)
    parts = load_era5(site)
    T_r, P_r = interp_at(parts, hours, z_t) if parts else (np.full(len(hours), np.nan),) * 2
    have_c = np.isfinite(T_c) & np.isfinite(P_c)
    T_e, P_e = np.where(have_c, T_c, T_r), np.where(have_c, P_c, P_r)
    have_e = np.isfinite(T_e) & np.isfinite(P_e)
    msrc = np.where(have_c, "cloudnet_model", "era5")
    sfc, ssrc = load_surface(site, hours)
    if sfc is not None:
        Ts, RHs, Ps = sfc["T"].values, sfc["RH"].values, sfc["P"].values
        if kind == "insitu":
            dz = alt - MET_ALT.get(ssrc.replace("arm_", ""), alt)     # met below the in-cloud station
            T_s, P_s, z_lcl = surface_to_cloud_base(Ts, RHs, Ps, np.full(len(hours), dz))
            z_lcl = np.full(len(hours), np.nan)
        else:
            T_s, P_s, z_lcl = surface_to_cloud_base(Ts, RHs, Ps, cbh)
        have_s = np.isfinite(T_s) & np.isfinite(P_s)
    else:
        T_s = P_s = z_lcl = np.full(len(hours), np.nan)
        have_s = np.zeros(len(hours), bool)
    dcoup = cbh - z_lcl
    coupled = np.abs(dcoup) <= coupled_m
    print(f"   surface met ({ssrc or 'none'}): {have_s.sum()} hours; model: {have_e.sum()} hours "
          f"(Cloudnet model {have_c.sum()}, ERA5 {int((have_e & ~have_c).sum())})"
          + ("" if kind == "insitu" or not have_s.any() else
             f"; cloud base - LCL median {np.nanmedian(dcoup):+.0f} m, coupled (|d| <= {coupled_m:.0f} m) "
             f"{int(np.sum(coupled & have_s))} of {have_s.sum()}"))

    src = ds["thermo_source"].values.astype(object).copy()
    meas = np.isin(src, MEASURED)
    T0, P0 = ds["T0"].values.copy(), ds["P0"].values.copy()
    if kind == "insitu":
        use_s, use_e = ~meas & have_s, ~meas & ~have_s & have_e
        tag_s, tag_e = "station_met", np.char.add(msrc.astype(str), "_station")
    else:
        use_s = ~meas & have_s & (coupled | ~have_e)
        use_e = ~meas & ~use_s & have_e
        tag_s = np.where(coupled, "surface_dry_adiabat", "surface_decoupled")
        tag_e = np.char.add(msrc.astype(str), np.where(have_s, "_decoupled", "_cloud_base"))
    T0[use_s], P0[use_s] = T_s[use_s], P_s[use_s]
    T0[use_e], P0[use_e] = T_e[use_e], P_e[use_e]
    src[use_s] = tag_s if isinstance(tag_s, str) else tag_s[use_s]
    src[use_e] = tag_e if isinstance(tag_e, str) else tag_e[use_e]

    both = have_s & have_e & (coupled if kind != "insitu" else True)
    if both.sum() >= 10:
        print(f"   surface - model ({int(both.sum())} h): T {np.nanmedian(T_s[both] - T_e[both]):+.1f} K "
              f"(IQR {np.nanpercentile(T_s[both] - T_e[both], 25):+.1f}..{np.nanpercentile(T_s[both] - T_e[both], 75):+.1f}), "
              f"p {np.nanmedian(P_s[both] - P_e[both]) / 100:+.0f} hPa")
    mb = (ds["thermo_source"].values == "built_file_cloud_base")
    for name, TT, PP, hv in (("surface", T_s, P_s, have_s & coupled), ("model", T_e, P_e, have_e)):
        m = mb & hv
        if m.sum() >= 10:
            print(f"   {name} - measured cloud base ({m.sum()} h): T {np.nanmedian(TT[m] - ds['T0'].values[m]):+.1f} K, "
                  f"p {np.nanmedian(PP[m] - ds['P0'].values[m]) / 100:+.0f} hPa")
    print("   T0/P0 sources: " + ", ".join(f"{k} {v}" for k, v in pd.Series(src).value_counts().items()))
    if dry_run:
        return
    ds["T0"] = ("hour", T0, ds["T0"].attrs)
    ds["P0"] = ("hour", P0, ds["P0"].attrs)
    ds["thermo_source"] = ("hour", src)
    ds["T_model"] = ("hour", T_e, {"units": "K", "long_name": "Cloudnet model (else ERA5) temperature at z_target"})
    ds["P_model"] = ("hour", P_e, {"units": "Pa", "long_name": "Cloudnet model (else ERA5) pressure at z_target"})
    ds["T_surface_cb"] = ("hour", T_s, {"units": "K", "long_name": "surface T on a dry adiabat to cloud base (station T in cloud)"})
    ds["P_surface_cb"] = ("hour", P_s, {"units": "Pa"})
    ds["cbh_minus_lcl"] = ("hour", dcoup, {"units": "m", "long_name": "cloud base minus surface LCL (coupling)"})
    ds["z_target"] = ("hour", z_t, {"units": "m a.s.l.", "long_name": "height of the ERA5 look-up"})
    tmp = path + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)
    print(f"   -> {os.path.relpath(path, HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--coupled", type=float, default=300.0,
                    help="max |cloud base - LCL| (m) for using the surface dry adiabat when ERA5 exists")
    a = ap.parse_args()
    sites = a.sites or sorted(os.path.basename(p).replace("_parcel_inputs.nc", "")
                              for p in glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))
    for s in sites:
        build_site(s, a.dry_run, a.coupled)


if __name__ == "__main__":
    main()
