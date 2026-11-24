#!/usr/bin/env python
"""Download ERA5 hourly temperature and geopotential on pressure levels for the cloud days of every site.

Used by build_thermo.py to get cloud-base temperature and pressure where the observations do not have them
(Cloudnet sites, in-cloud stations). One request per ERA5 grid cell and month, cloud days only, all 24 hours,
a 0.5 deg box (3 x 3 points) and the pressure levels from just below the station to ~4 km above it.
Sites in the same ERA5 cell (ENA and its Cloudnet variant, the two EPCAPE sites, ...) share files:
data/era5/<cell>/era5_<cell>_<YYYY-MM>.nc.

ERA5 is on the Climate Data Store (CDS), not the Atmosphere Data Store used for CAMS. Put the CDS
url and key in ~/.cdsapirc_cds (two lines, url: https://cds.climate.copernicus.eu/api and key: <your CDS
token>) and accept the ERA5 pressure-levels licence on the CDS website. The script points cdsapi at that
file, so ~/.cdsapirc (ADS) stays as it is for the CAMS downloads.

Examples
--------
python era5_download.py --dry-run
nohup python -u era5_download.py > logs/era5.log 2>&1 &
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sites import SITE_COORDS  # noqa: E402

LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500]
DATASET = "reanalysis-era5-pressure-levels"


def cell_of(site):
    lat, lon, _ = SITE_COORDS[site]
    return round(lat * 4) / 4, round(lon * 4) / 4


def cell_name(cell):
    lat, lon = cell
    return f"{'n' if lat >= 0 else 's'}{abs(lat):.2f}_{'e' if lon >= 0 else 'w'}{abs(lon):.2f}"


def site_levels(alt_m, above_m=4000.0):
    p_lo = 1013.25 * (1 - 2.25577e-5 * max(alt_m - 300.0, 0)) ** 5.25588     # a little below the station
    p_hi = 1013.25 * (1 - 2.25577e-5 * (alt_m + above_m)) ** 5.25588
    lev = [p for p in LEVELS if p_hi - 25 <= p <= p_lo + 25]
    return lev or LEVELS


def cloud_days(site):
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(p):
        return set()
    with xr.open_dataset(p) as ds:
        return set(pd.DatetimeIndex(ds["hour"].values).normalize())


def plan(sites_filter=None):
    sites = sorted(os.path.basename(p).replace("_parcel_inputs.nc", "")
                   for p in glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))
    cells = defaultdict(lambda: dict(sites=[], days=set(), levels=set()))
    for s in sites:
        if sites_filter and s not in sites_filter:
            continue
        c = cells[cell_of(s)]
        c["sites"].append(s)
        c["days"] |= cloud_days(s)
        c["levels"] |= set(site_levels(SITE_COORDS[s][2]))
    jobs, notes = [], []
    for cell, c in sorted(cells.items(), key=lambda kv: cell_name(kv[0])):
        name = cell_name(cell)
        lat, lon = cell
        by_month = defaultdict(list)
        for d in sorted(c["days"]):
            by_month[d.strftime("%Y-%m")].append(d)
        lev = [str(p) for p in sorted(c["levels"], reverse=True)]
        for m, dl in sorted(by_month.items()):
            req = {"product_type": ["reanalysis"], "variable": ["temperature", "geopotential"],
                   "year": [m[:4]], "month": [m[5:]], "day": sorted({f"{d.day:02d}" for d in dl}),
                   "time": [f"{h:02d}:00" for h in range(24)], "pressure_level": lev,
                   "area": [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],
                   "data_format": "netcdf", "download_format": "unarchived"}
            out = os.path.join(HERE, "data", "era5", name, f"era5_{name}_{m}.nc")
            jobs.append((name, m, req, out, DATASET))
        notes.append(f"{name:18s} {','.join(c['sites']):35s} {len(c['days']):5d} days, {len(by_month):4d} months, "
                     f"levels {lev[0]}-{lev[-1]} hPa")
    return jobs, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+")
    ap.add_argument("--workers", type=int, default=2, help="requests in the CDS queue at the same time")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--rc", default=os.path.expanduser("~/.cdsapirc_cds"), help="cdsapi rc file for the CDS")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs, notes = plan(a.sites)
    print("\n".join(notes))
    todo = [j for j in jobs if not os.path.exists(j[3])]
    print(f"\n{len(jobs)} requests, {len(jobs) - len(todo)} already done, {len(todo)} to run ({a.workers} at a time)")
    if a.dry_run or not todo:
        if todo:
            print("example request:", todo[0][2])
        return
    if not os.path.exists(a.rc):
        raise SystemExit(f"no {a.rc}: create it with the CDS url and key (see the docstring)")
    os.environ["CDSAPI_RC"] = a.rc
    from cams_download import log, run_job                                           # same queue handling
    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(run_job, j, a.retries) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            ok += bool(f.result())
            log(f"progress {k}/{len(todo)} finished, {ok} ok")


if __name__ == "__main__":
    main()
