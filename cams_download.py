#!/usr/bin/env python
"""Download CAMS global reanalysis (EAC4) aerosol species for the cloud hours of all sites.

Sites are grouped into regions (one box each), because ADS retrieves whole global fields from tape
whatever the area: one request for a box around all European sites costs about the same as one for
a single site. Two modes:

  daily    (default) 3-hourly EAC4, one request per region and month, only for the days with cloud
           hours at any site in the region. Written to data/cams/<region>/eac4_<region>_<YYYY-MM>.nc.
  monthly  EAC4 monthly means (separate, much faster dataset), one request per region and year.
           Written to data/cams_monthly/<region>/eac4m_<region>_<YYYY>.nc.

Variables: sulphate, hydrophilic + hydrophobic organic matter, hydrophilic + hydrophobic black carbon,
sea salt 0.03-0.5 / 0.5-5 / 5-20 um, dust 0.03-0.55 / 0.55-0.9 / 0.9-20 um (kg kg-1). EAC4 has no
nitrate or ammonium. Temperature and humidity come from ERA5 in the thermodynamics step instead.

ADS allows only about one queued request per dataset and user, so requests run one at a time (a
"queue full" reply makes the script wait 10 min and resubmit). The two modes use different datasets
and can run at the same time. Finished files are skipped, so the script can be stopped and restarted.
Months after --last-month (EAC4's current end) are listed only. Killing the script does not cancel a
request already queued at ADS: cancel stale ones at https://ads.atmosphere.copernicus.eu/requests.

Needs ~/.cdsapirc for https://ads.atmosphere.copernicus.eu/api, the EAC4 licences accepted, and
cdsapi >= 0.7.2. Run on a login node (compute nodes may have no internet) under nohup or tmux.

Examples
--------
python cams_download.py --dry-run                         # requests per region, daily mode
python cams_download.py --mode monthly --dry-run
nohup python -u cams_download.py --mode monthly > logs/cams_monthly.log 2>&1 &
nohup python -u cams_download.py --mode daily   > logs/cams_daily.log 2>&1 &
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sites import SITE_COORDS  # noqa: E402

VARIABLES = [
    "sulphate_aerosol_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio", "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydrophilic_black_carbon_aerosol_mixing_ratio", "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "sea_salt_aerosol_0.03-0.5um_mixing_ratio", "sea_salt_aerosol_0.5-5um_mixing_ratio",
    "sea_salt_aerosol_5-20um_mixing_ratio",
    "dust_aerosol_0.03-0.55um_mixing_ratio", "dust_aerosol_0.55-0.9um_mixing_ratio",
    "dust_aerosol_0.9-20um_mixing_ratio",
]
PLEVS = [1000, 950, 925, 900, 850, 800, 700, 600]   # hPa; EAC4 pressure levels in the lowest ~4 km
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]

# Regions: (N, W, S, E) boxes, each containing the listed sites with a margin of >= 0.75 deg.
REGIONS = {
    "europe": ((80.5, -11.0, 36.0, 28.5),
               ["puijo", "pallas", "kenttarova", "zeppelin", "ny-alesund", "hyytiala", "mace-head", "cabauw",
                "melpitz", "palaiseau", "bucharest", "granada", "schneefernerhaus", "norunda"]),
    "sgp_bnf": ((37.5, -98.5, 33.5, -86.5), ["sgp", "sgp_cloudnet", "bankhead"]),
    "epcape": ((33.75, -118.0, 32.0, -116.5), ["epcape_pier", "epcape_soledad"]),
    "ena": ((40.0, -29.0, 38.25, -27.0), ["ena", "ena_cloudnet"]),
    "lasic": ((-7.0, -15.25, -8.75, -13.5), ["lasic", "lasic_cloudnet"]),
}


def site_levels(alt_m, depth_hpa=200, below_hpa=30):
    """Pressure levels from a little below the station's standard-atmosphere pressure to ~2 km above."""
    p_site = 1013.25 * np.exp(-alt_m / 8000.0)
    return [p for p in PLEVS if p_site - depth_hpa <= p <= p_site + below_hpa]


def cloud_hours(site):
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(p):
        return None
    with xr.open_dataset(p) as ds:
        return pd.DatetimeIndex(ds["hour"].values)


def plan(mode, last_month, sites_filter=None):
    """List of (region, key, request, outfile) and per-region notes."""
    last = pd.Period(last_month, "M")
    jobs, notes = [], []
    for region, (box, members) in REGIONS.items():
        members = [s for s in members if (not sites_filter or s in sites_filter)]
        days, levels, used, beyond = set(), set(), [], defaultdict(int)
        for s in members:
            h = cloud_hours(s)
            if h is None:
                continue
            used.append(s)
            levels |= set(site_levels(SITE_COORDS[s][2]))
            for d in h.normalize().unique():
                if pd.Period(d, "M") > last:
                    beyond[s] += 1
                else:
                    days.add(d)
        if not used:
            continue
        lev = [str(p) for p in sorted(levels, reverse=True)]
        base = {"variable": VARIABLES, "pressure_level": lev, "area": list(box), "data_format": "netcdf"}
        days = sorted(days)
        if mode == "daily":
            by_month = defaultdict(list)
            for d in days:
                by_month[d.strftime("%Y-%m")].append(d.strftime("%Y-%m-%d"))
            for m, dl in sorted(by_month.items()):
                out = os.path.join(HERE, "data", "cams", region, f"eac4_{region}_{m}.nc")
                jobs.append((region, m, dict(base, date=dl, time=TIMES), out, "cams-global-reanalysis-eac4"))
            n = len(by_month)
        else:
            by_year = defaultdict(set)
            for d in days:
                by_year[d.year].add(f"{d.month:02d}")
            for y, ms in sorted(by_year.items()):
                out = os.path.join(HERE, "data", "cams_monthly", region, f"eac4m_{region}_{y}.nc")
                jobs.append((region, str(y), dict(base, year=[str(y)], month=sorted(ms),
                                                  product_type=["monthly_mean"]),
                             out, "cams-global-reanalysis-eac4-monthly"))
            n = len(by_year)
        notes.append(f"{region:8s} {len(used):2d} sites, {len(days):5d} cloud days, levels {','.join(lev)} hPa -> "
                     f"{n} requests" + (f"  | days after EAC4: " + ", ".join(f"{k} {v}" for k, v in beyond.items())
                                        if beyond else ""))
    return jobs, notes


_print = threading.Lock()


def log(msg):
    with _print:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def run_job(job, retries):
    import cdsapi
    region, key, req, out, dataset = job
    os.makedirs(os.path.dirname(out), exist_ok=True)
    part = out + ".part"
    client = cdsapi.Client()
    attempt, waited = 0, 0
    while attempt < retries:
        attempt += 1
        t0 = time.time()
        try:
            log(f"submit {region} {key} (attempt {attempt}, {len(req.get('date', req.get('month', [])))} "
                f"{'days' if 'date' in req else 'months'})")
            client.retrieve(dataset, req).download(part)
            os.replace(part, out)
            log(f"done   {region} {key} in {(time.time() - t0) / 60:.0f} min")
            return True
        except Exception as e:                                                   # noqa: BLE001
            msg = str(e).replace("\n", " ")[:300]
            log(f"failed {region} {key}: {msg}")
            if os.path.exists(part):
                os.remove(part)
            low = msg.lower()
            if "maximum number queued" in low or "too many" in low:
                # ADS per-user queue limit for this dataset: wait for a slot, not counted as an attempt
                attempt -= 1
                waited += 1
                if waited > 144:                       # ~1 day of waiting
                    log(f"giving up {region} {key}: queue still full after a day")
                    return False
                time.sleep(600)
                continue
            if "licence" in low or "license" in low:
                log("   accept the licence for this dataset on the ADS website (Download tab), then rerun")
                return False
            if any(k in low for k in ("no data", "not available", "valid combination")):
                return False
            time.sleep(120 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["daily", "monthly"], default="daily")
    ap.add_argument("--sites", nargs="+", help="restrict to these sites (default: all with inputs)")
    ap.add_argument("--workers", type=int, default=1,
                    help="requests in the ADS queue at the same time (ADS allows ~1 per dataset and user)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--last-month", default="2025-12", help="last month EAC4 covers on the ADS")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs, notes = plan(a.mode, a.last_month, a.sites)
    print("\n".join(notes))
    todo = [j for j in jobs if not os.path.exists(j[3])]
    print(f"\n{len(jobs)} requests, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"({a.mode} mode, {a.workers} at a time)")
    if a.dry_run or not todo:
        if todo:
            r = dict(todo[0][2])
            if "date" in r:
                r["date"] = r["date"][:3] + ["..."]
            print("example request:", todo[0][4], r)
        return
    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(run_job, j, a.retries) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            ok += bool(f.result())
            log(f"progress {k}/{len(todo)} finished, {ok} ok")


if __name__ == "__main__":
    main()
