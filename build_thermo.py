#!/usr/bin/env python
"""Cloud-base temperature and pressure (parcel start T0, P0) from ERA5 for hours without measured values.

For each hour of inputs/<site>_parcel_inputs.nc, ERA5 temperature and pressure are interpolated (T linear,
ln p linear in geopotential height) at the nearest grid point to

  site altitude + cloud_base_height     Cloudnet and NDROP sites (cloud base height above ground)
  site altitude                         in-cloud stations (kind "insitu": Puijo, Pallas, Mt Soledad, Zeppelin)

The ERA5 values are always stored (T_era5, P_era5, z_target). T0/P0 are replaced only where
thermo_source is still "placeholder"; measured values (built_file_cloud_base at the NDROP sites,
zeppelin_station) are kept. For the NDROP sites the script prints ERA5 minus measured, with the cloud base
taken above ground and above sea level, as a check of both ERA5 and the height convention.

Needs data/era5/ from era5_download.py.

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


def build_site(site, dry_run=False):
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
    parts = load_era5(site)
    if not parts:
        print(f"   no ERA5 files in data/era5/{cell_name(cell_of(site))}/")
        return
    T, P = interp_at(parts, hours, z_t)
    have = np.isfinite(T) & np.isfinite(P)
    print(f"   ERA5 for {have.sum()} hours: T {np.nanmedian(T) - 273.15:.1f} C, p {np.nanmedian(P) / 100:.0f} hPa (medians)")

    src = ds["thermo_source"].values.astype(object).copy()
    meas = src != "placeholder"
    if (src == "built_file_cloud_base").any():
        m = (src == "built_file_cloud_base") & have
        Ta, Pa = interp_at(parts, hours, cbh)               # cloud base taken as above sea level instead
        dT, dP = T[m] - ds["T0"].values[m], (P[m] - ds["P0"].values[m]) / 100
        dPa = (Pa[m] - ds["P0"].values[m]) / 100
        print(f"   ERA5 - measured cloud base ({m.sum()} h): T {np.nanmedian(dT):+.1f} K "
              f"(IQR {np.nanpercentile(dT, 25):+.1f}..{np.nanpercentile(dT, 75):+.1f}), "
              f"p {np.nanmedian(dP):+.0f} hPa above ground / {np.nanmedian(dPa):+.0f} hPa if a.s.l.")
    use = ~meas & have
    T0, P0 = ds["T0"].values.copy(), ds["P0"].values.copy()
    T0[use], P0[use], src[use] = T[use], P[use], "era5_cloud_base" if kind != "insitu" else "era5_station"
    left = (src == "placeholder").sum()
    print(f"   T0/P0: {use.sum()} hours from ERA5, {meas.sum()} measured kept, {left} still placeholder")
    if dry_run:
        return
    ds["T0"] = ("hour", T0, ds["T0"].attrs)
    ds["P0"] = ("hour", P0, ds["P0"].attrs)
    ds["thermo_source"] = ("hour", src)
    ds["T_era5"] = ("hour", T, {"units": "K", "long_name": "ERA5 temperature at z_target"})
    ds["P_era5"] = ("hour", P, {"units": "Pa", "long_name": "ERA5 pressure at z_target"})
    ds["z_target"] = ("hour", z_t, {"units": "m a.s.l.", "long_name": "height of the ERA5 look-up"})
    tmp = path + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)
    print(f"   -> {os.path.relpath(path, HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sites = a.sites or sorted(os.path.basename(p).replace("_parcel_inputs.nc", "")
                              for p in glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))
    for s in sites:
        build_site(s, a.dry_run)


if __name__ == "__main__":
    main()
