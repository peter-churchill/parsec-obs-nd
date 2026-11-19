#!/usr/bin/env python
"""Download CAMS global reanalysis (EAC4) aerosol species around each site, for the months with cloud hours.

One request per site and month, over a small box around the site (EAC4 is on a 0.75 deg grid), on the
pressure levels from just above the surface to about 2 km above it. Written to
data/cams/<site>/eac4_<site>_<YYYY-MM>.nc; months already downloaded are skipped, so the script can be
stopped and restarted.

Variables (mass mixing ratios, kg kg-1, plus the state needed to place them in the vertical):
    sulphate, hydrophilic + hydrophobic organic matter, hydrophilic + hydrophobic black carbon,
    sea salt 0.03-0.5 / 0.5-5 / 5-20 um, dust 0.03-0.55 / 0.55-0.9 / 0.9-20 um,
    temperature, specific humidity, geopotential.
EAC4 has no nitrate or ammonium.

Needs ~/.cdsapirc pointing at https://ads.atmosphere.copernicus.eu/api, the EAC4 licence accepted on the
ADS website, and cdsapi >= 0.7.2. Run it on a login node (compute nodes may have no internet), e.g. in
tmux or with nohup, since ADS queues requests and a site with many months takes hours.

Examples
--------
python cams_download.py --dry-run                      # list the requests
python cams_download.py --sites hyytiala               # one site first
nohup python -u cams_download.py > logs/cams_download.log 2>&1 &
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sites import SITE_COORDS  # noqa: E402

DATASET = "cams-global-reanalysis-eac4"
VARIABLES = [
    "sulphate_aerosol_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio", "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydrophilic_black_carbon_aerosol_mixing_ratio", "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "sea_salt_aerosol_0.03-0.5um_mixing_ratio", "sea_salt_aerosol_0.5-5um_mixing_ratio",
    "sea_salt_aerosol_5-20um_mixing_ratio",
    "dust_aerosol_0.03-0.55um_mixing_ratio", "dust_aerosol_0.55-0.9um_mixing_ratio",
    "dust_aerosol_0.9-20um_mixing_ratio",
    "temperature", "specific_humidity", "geopotential",
]
PLEVS = [1000, 950, 925, 900, 850, 800, 700, 600]   # hPa; the EAC4 pressure levels in the lowest ~4 km
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]


def site_levels(alt_m, depth_hpa=200, below_hpa=30):
    """Pressure levels from a little below the station's standard-atmosphere pressure to ~2 km above."""
    p_site = 1013.25 * np.exp(-alt_m / 8000.0)
    return [str(p) for p in PLEVS if p_site - depth_hpa <= p <= p_site + below_hpa]


def months_with_cloud(site):
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(p):
        return []
    with xr.open_dataset(p) as ds:
        h = pd.DatetimeIndex(ds["hour"].values)
    return sorted(set(h.to_period("M")))


def request_for(site, month, box):
    lat, lon, alt = SITE_COORDS[site]
    first = month.start_time.strftime("%Y-%m-%d")
    last = month.end_time.strftime("%Y-%m-%d")
    return {
        "variable": VARIABLES,
        "pressure_level": site_levels(alt),
        "date": [f"{first}/{last}"],
        "time": TIMES,
        "area": [round(lat + box, 2), round(lon - box, 2), round(lat - box, 2), round(lon + box, 2)],  # N W S E
        "data_format": "netcdf",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", help="default: every site with an inputs file")
    ap.add_argument("--outdir", default=os.path.join(HERE, "data", "cams"))
    ap.add_argument("--box", type=float, default=0.75, help="half-width of the box around the site (deg)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--last-month", default="2025-12",
                    help="last month EAC4 covers on the ADS (later months are listed, not requested)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sites = a.sites or sorted(f[:-len("_parcel_inputs.nc")] for f in os.listdir(os.path.join(HERE, "inputs"))
                              if f.endswith("_parcel_inputs.nc"))
    todo = []
    for s in sites:
        if s not in SITE_COORDS:
            print(f"{s}: no coordinates in sites.SITE_COORDS, skipped")
            continue
        months = months_with_cloud(s)
        last = pd.Period(a.last_month, "M")
        beyond = [m for m in months if m > last]
        months = [m for m in months if m <= last]
        missing = [m for m in months
                   if not os.path.exists(os.path.join(a.outdir, s, f"eac4_{s}_{m}.nc"))]
        print(f"{s:18s} {len(months):3d} months with cloud hours, {len(missing):3d} to download, "
              f"levels {','.join(site_levels(SITE_COORDS[s][2]))} hPa"
              + (f"  | {len(beyond)} months after EAC4 ({beyond[0]}..{beyond[-1]}): need the CAMS analysis"
                 if beyond else ""))
        todo += [(s, m) for m in missing]
    print(f"\n{len(todo)} requests in total")
    if a.dry_run or not todo:
        if todo:
            print("example request:", request_for(*todo[0], a.box))
        return

    import cdsapi
    client = cdsapi.Client()
    t0 = time.time()
    for k, (s, m) in enumerate(todo, 1):
        out = os.path.join(a.outdir, s, f"eac4_{s}_{m}.nc")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        part = out + ".part"
        for attempt in range(1, a.retries + 1):
            try:
                print(f"[{k}/{len(todo)}] {s} {m} (attempt {attempt}) ...", flush=True)
                client.retrieve(DATASET, request_for(s, m, a.box)).download(part)
                os.replace(part, out)
                break
            except Exception as e:                                                   # noqa: BLE001
                msg = str(e).replace("\n", " ")[:300]
                print(f"   failed: {msg}", flush=True)
                if os.path.exists(part):
                    os.remove(part)
                if "licence" in msg.lower() or "license" in msg.lower():
                    sys.exit("Accept the EAC4 licence on the ADS website (dataset page, Download tab) and rerun.")
                if any(k in msg.lower() for k in ("no data", "not available", "valid combination")):
                    print("   (month outside EAC4 coverage? skipping)")
                    break
                time.sleep(60 * attempt)
        el = (time.time() - t0) / 3600
        print(f"   done {k}/{len(todo)}, {el:.1f} h elapsed", flush=True)


if __name__ == "__main__":
    main()
