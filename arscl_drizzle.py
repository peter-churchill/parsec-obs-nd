#!/usr/bin/env python
"""Drizzle below cloud base from ARM's KAZR ARSCL, for the ARM sites (ENA, SGP, LASIC, ...).

Cloudnet's arm-graciosa categorize files stop in 2019 and restart in 2026, so the ENA parcel hours (2022-2025) have
no Cloudnet drizzle flag. ARSCL (KAZR Active Remote Sensing of CLouds, Kollias version) has radar reflectivity
and a best-estimate cloud base on a ~4-s grid.

Per profile with a cloud base below --max-base m:
  drizzle   max reflectivity in the layer --below-top to --below-bottom m below cloud base > --dbz (default
            -20 dBZ, the usual drizzle threshold: cloud droplets alone stay below about -20 dBZ)
  zmax      max reflectivity from cloud base to 1 km above it (column drizzle intensity)
Hourly: f_drizzle = fraction of cloudy profiles with drizzle below base, zmax_cloud_dbz = median zmax.

Written into results/cloudnet_quality/<site>_flags.csv (created if absent): f_drizzle is filled where the
Cloudnet flag is missing, and f_drizzle_arscl / zmax_cloud_dbz are added; nd_ratio_drivers.py then uses them.

Needs ARM_USER / ARM_TOKEN and OBS_REPO (for arm_query / arm_download).

python arscl_drizzle.py download --sites ena --dry-run       # files and size
python arscl_drizzle.py download --sites ena
python arscl_drizzle.py flags --sites ena
python nd_ratio_drivers.py --sites ena
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "data", "arscl")
FLAGS = os.path.join(HERE, "results", "cloudnet_quality")
# ARSCL KAZR (Kollias) datastreams; the first one that lists files for the site's days is used
STREAMS = {
    "ena": ["enaarsclkazr1kolliasC1.c0", "enaarsclkazr1kolliasC1.c1"],
    "ena_cloudnet": ["enaarsclkazr1kolliasC1.c0", "enaarsclkazr1kolliasC1.c1"],
    "sgp": ["sgparsclkazr1kolliasC1.c0", "sgparsclkazr1kolliasC1.c1"],
    "sgp_cloudnet": ["sgparsclkazr1kolliasC1.c0", "sgparsclkazr1kolliasC1.c1"],
    "lasic": ["asiarsclkazr1kolliasM1.c0", "asiarsclkazr1kolliasM1.c1"],
    "epcape_pier": ["epcarsclkazr1kolliasM1.c0", "epcarsclkazr1kolliasM1.c1"],
    "bankhead": ["bnfarsclkazr1kolliasM1.c0", "bnfarsclkazr1kolliasM1.c1"],
}


def hours_of(site):
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    with xr.open_dataset(p) as ds:
        return pd.DatetimeIndex(ds["hour"].values)


def arm_tools(obs_repo):
    if not obs_repo:
        raise SystemExit("set OBS_REPO (arm_ena_nd_pnsd.py has arm_query / arm_download)")
    sys.path.insert(0, obs_repo)
    from arm_ena_nd_pnsd import _file_date, arm_download, arm_query
    return arm_query, arm_download, _file_date


def download(a):
    arm_query, arm_download, _file_date = arm_tools(a.obs_repo)
    user, token = os.environ.get("ARM_USER", "").strip(), os.environ.get("ARM_TOKEN", "").strip()
    if not user or not token:
        raise SystemExit("set ARM_USER and ARM_TOKEN")
    for site in a.sites:
        days = set(hours_of(site).normalize())
        start, end = min(days).strftime("%Y-%m-%d"), (max(days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for stream in STREAMS.get(site, []):
            files = [f for f in arm_query(user, token, stream, start, end)
                     if _file_date(f) is not None and _file_date(f).normalize() in days]
            print(f"{site:14s} {stream:30s} {len(files):4d} files for {len(days)} cloud days", flush=True)
            if not files:
                continue
            if not a.dry_run:
                outdir = os.path.join(OUT, stream)
                os.makedirs(outdir, exist_ok=True)
                res = [arm_download(user, token, f, outdir) for f in files]
                print(f"    downloaded {res.count('downloaded')}, failed {res.count('failed')}", flush=True)
            break


def var(ds, *names, contains=None):
    for n in names:
        if n in ds:
            return ds[n]
    if contains:
        for n in ds.data_vars:
            if contains in n and not n.startswith("qc_"):
                return ds[n]
    raise KeyError(f"none of {names} / *{contains}* in file")


def flags_site(site, a):
    hours = hours_of(site)
    days = set(hours.normalize())
    files = []
    for stream in STREAMS.get(site, []):
        files += sorted(glob.glob(os.path.join(OUT, stream, "*.nc")))
    rows = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                t = pd.DatetimeIndex(ds["time"].values)
                if t.normalize()[0] not in days and t.normalize()[-1] not in days:
                    continue
                hr = t.floor("h")
                keep = np.asarray(hr.isin(hours))
                if not keep.any():
                    continue
                z = var(ds, "height").values.astype(float)
                if np.nanmedian(z) > 5000:                       # heights given above sea level
                    z = z - float(ds["alt"].values) if "alt" in ds else z
                ref = var(ds, "reflectivity_best_estimate", "reflectivity", contains="reflectivity").values[keep]
                cb = var(ds, "cloud_base_best_estimate", contains="cloud_base").values[keep].astype(float)
                if cb.ndim > 1:
                    cb = cb[:, 0]
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:90]}")
            continue
        cb = np.where((cb > 0) & (cb < a.max_base), cb, np.nan)
        h = z[None, :]
        c = cb[:, None]
        below = (h >= c - a.below_bottom) & (h <= c - a.below_top)
        incl = (h >= c) & (h <= c + 1000.0)
        with np.errstate(all="ignore"):
            zb = np.nanmax(np.where(below, ref, np.nan), axis=1)
            zc = np.nanmax(np.where(incl, ref, np.nan), axis=1)
        cloudy = np.isfinite(cb)
        rows.append(pd.DataFrame(dict(hour=hr[keep], cloudy=cloudy, drizzle=cloudy & (zb > a.dbz),
                                      zmax=np.where(cloudy, zc, np.nan))))
    if not rows:
        print(f"{site}: no ARSCL files with its hours (run download first)")
        return None
    df = pd.concat(rows)
    g = df.groupby("hour")
    nc = g["cloudy"].sum().replace(0, np.nan)
    out = pd.DataFrame({"n_cloudy_arscl": g["cloudy"].sum(), "f_drizzle_arscl": g["drizzle"].sum() / nc,
                        "zmax_cloud_dbz": g["zmax"].median()}).reindex(hours)
    return out


def flags(a):
    os.makedirs(FLAGS, exist_ok=True)
    for site in a.sites:
        f = flags_site(site, a)
        if f is None:
            continue
        p = os.path.join(FLAGS, f"{site}_flags.csv")
        old = pd.read_csv(p, parse_dates=["hour"]).set_index("hour") if os.path.exists(p) else pd.DataFrame(index=f.index)
        old = old.drop(columns=[c for c in f.columns if c in old], errors="ignore")
        d = old.join(f, how="outer")
        if "f_drizzle" not in d:
            d["f_drizzle"] = np.nan
        d["f_drizzle"] = d["f_drizzle"].fillna(d["f_drizzle_arscl"])
        d.index.name = "hour"
        d.to_csv(p, float_format="%.4g")
        ok = f["f_drizzle_arscl"].notna()
        print(f"{site:14s} ARSCL for {ok.sum()} of {len(f)} hours; f_drizzle median {f['f_drizzle_arscl'].median():.2f}, "
              f"hours with > 50 % drizzle {(f['f_drizzle_arscl'] > 0.5).mean():.0%}; "
              f"in-cloud Zmax median {f['zmax_cloud_dbz'].median():.1f} dBZ -> {os.path.relpath(p, HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "flags"])
    ap.add_argument("--sites", nargs="+", default=["ena"])
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dbz", type=float, default=-20.0, help="drizzle threshold below cloud base (dBZ)")
    ap.add_argument("--below-top", type=float, default=50.0, help="layer top, m below cloud base")
    ap.add_argument("--below-bottom", type=float, default=300.0, help="layer bottom, m below cloud base")
    ap.add_argument("--max-base", type=float, default=3000.0, help="ignore cloud bases above this (m)")
    a = ap.parse_args()
    {"download": download, "flags": flags}[a.step](a)


if __name__ == "__main__":
    main()
