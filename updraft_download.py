#!/usr/bin/env python
"""Download Doppler-lidar vertical velocity for the cloud days of every site that has it.

ARM: the vertical-velocity statistics VAP (dlprofwstats4news), first stream with files per site:
  ENA (+ Cloudnet variant)  enadlprofwstats4newsC1.c1
  SGP (+ Cloudnet variant)  sgpdlprofwstats4newsC1.c1, gaps filled from E37
  LASIC (+ Cloudnet)        asidlprofwstats4newsM1.c1
  EPCAPE pier and Soledad   epcdlprofwstats4newsM1.c1 (the pier lidar sees cloud base from below; the Mt
                            Soledad lidar, S2, sits in the cloud when the station does)
  Bankhead                  bnfdlprofwstats4newsM1.c1
  -> data/updraft/arm/<stream>/
Cloudnet: the doppler-lidar product (vertical stare) for the sites updraft_search.py found it for
(Hyytiala, Kenttarova, Granada, Melpitz, Bucharest, Cabauw) -> data/updraft/cloudnet/<site id>/.
With --product categorize (or radar, epsilon-radar): cloud-radar data for the sites without lidar
(Norunda, Mace Head, Palaiseau, Schneefernerhaus, Ny-Alesund, and Ny-Alesund / Kenttarova for Zeppelin /
Pallas) -> data/updraft/cloudnet_<product>/<site id>/

Files already present are skipped. Needs network, OBS_REPO and ARM_USER / ARM_TOKEN for ARM.

Examples
--------
python updraft_download.py --dry-run
python updraft_download.py --source cloudnet --product categorize --dry-run
nohup python -u updraft_download.py > logs/updraft_download.log 2>&1 &
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from obs_composition_download import cloud_days  # noqa: E402

OUT = os.path.join(HERE, "data", "updraft")
CLOUDNET = "https://cloudnet.fmi.fi/api"
ARM_STREAMS = {
    "ena": ["enadlprofwstats4newsC1.c1"], "ena_cloudnet": ["enadlprofwstats4newsC1.c1"],
    "sgp": ["sgpdlprofwstats4newsC1.c1", "sgpdlprofwstats4newsE37.c1"],
    "sgp_cloudnet": ["sgpdlprofwstats4newsC1.c1", "sgpdlprofwstats4newsE37.c1"],
    "lasic": ["asidlprofwstats4newsM1.c1"], "lasic_cloudnet": ["asidlprofwstats4newsM1.c1"],
    "epcape_pier": ["epcdlprofwstats4newsM1.c1"], "epcape_soledad": ["epcdlprofwstats4newsM1.c1"],
    "bankhead": ["bnfdlprofwstats4newsM1.c1"],
}
CLOUDNET_SITES = {"hyytiala": "hyytiala", "kenttarova": "kenttarova", "granada": "granada",
                  "melpitz": "melpitz", "bucharest": "bucharest", "cabauw": "cabauw"}
# sites without Doppler lidar on their cloud days: cloud radar Doppler velocity (Cloudnet categorize: radar v
# and Z on a 30-s grid with the target classification, much smaller than the level-1 radar files)
CLOUDNET_RADAR_SITES = {"norunda": "norunda", "mace-head": "mace-head", "palaiseau": "palaiseau",
                        "schneefernerhaus": "schneefernerhaus", "ny-alesund": "ny-alesund",
                        "zeppelin": "ny-alesund", "pallas": "kenttarova",
                        # lidar sites too: radar fills the hours without lidar and gives a lidar/radar comparison
                        "hyytiala": "hyytiala", "granada": "granada", "melpitz": "melpitz",
                        "bucharest": "bucharest", "cabauw": "cabauw", "kenttarova": "kenttarova"}


def arm(a):
    from arm_ena_nd_pnsd import _file_date, arm_download, arm_query
    user, token = os.environ.get("ARM_USER", "").strip(), os.environ.get("ARM_TOKEN", "").strip()
    if not user or not token:
        print("ARM: set ARM_USER and ARM_TOKEN; skipping")
        return
    want = {}                                                   # stream -> set of days
    for site, streams in ARM_STREAMS.items():
        days = cloud_days(site)
        if a.sites and site not in a.sites or not days:
            continue
        for s in streams:
            want.setdefault(s, set()).update(days)
    covered = {}
    for stream, days in want.items():
        start = min(days).strftime("%Y-%m-%d")
        end = (max(days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        files = arm_query(user, token, stream, start, end)
        group = stream[:3]
        # E37 only for days the C1 stream does not have
        files = [f for f in files if _file_date(f) is not None and _file_date(f).normalize() in days
                 and _file_date(f).normalize() not in covered.get(group, set())]
        covered.setdefault(group, set()).update(_file_date(f).normalize() for f in files)
        outdir = os.path.join(OUT, "arm", stream)
        print(f"ARM {stream:30s} {len(files):4d} files", flush=True)
        if a.dry_run:
            continue
        os.makedirs(outdir, exist_ok=True)
        res = [arm_download(user, token, f, outdir) for f in files]
        print(f"    downloaded {res.count('downloaded')}, failed {res.count('failed')}", flush=True)


def get_json(url, params):
    with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params), timeout=120) as r:
        return json.loads(r.read().decode())


def fetch(url, dest, retries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped"
    for k in range(retries):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, dest)
            return "downloaded"
        except Exception as e:                                                      # noqa: BLE001
            if k == retries - 1:
                print(f"    failed {os.path.basename(dest)}: {str(e)[:100]}")
            time.sleep(10 * (k + 1))
    return "failed"


def cloudnet(a):
    table = CLOUDNET_SITES if a.product == "doppler-lidar" else CLOUDNET_RADAR_SITES
    seen = set()
    for site, cid in table.items():
        days = cloud_days(site)
        if a.sites and site not in a.sites or not days:
            continue
        files = get_json(f"{CLOUDNET}/files", {"site": cid, "product": a.product,
                                               "dateFrom": min(days).strftime("%Y-%m-%d"),
                                               "dateTo": max(days).strftime("%Y-%m-%d")})
        files = [f for f in files if pd.Timestamp(f["measurementDate"]) in days and f["filename"] not in seen]
        seen |= {f["filename"] for f in files}
        mb = sum(float(f.get("size", 0)) for f in files) / 1e6
        print(f"Cloudnet {a.product} {site:16s} ({cid:16s}) {len(files):4d} files, {mb:7.0f} MB", flush=True)
        if a.dry_run:
            continue
        outdir = os.path.join(OUT, "cloudnet" if a.product == "doppler-lidar" else f"cloudnet_{a.product}", cid)
        os.makedirs(outdir, exist_ok=True)
        res = [fetch(f["downloadUrl"], os.path.join(outdir, f["filename"])) for f in files]
        print(f"    downloaded {res.count('downloaded')}, skipped {res.count('skipped')}, "
              f"failed {res.count('failed')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--source", choices=["arm", "cloudnet", "both"], default="both")
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--product", default="doppler-lidar",
                    help="Cloudnet product: doppler-lidar (lidar sites) or categorize / radar / epsilon-radar for the "
                         "sites without lidar (CLOUDNET_RADAR_SITES)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.source in ("arm", "both"):
        if not a.obs_repo:
            ap.error("give --obs-repo or set OBS_REPO for ARM, or --source cloudnet")
        sys.path.insert(0, a.obs_repo)
        arm(a)
    if a.source in ("cloudnet", "both"):
        cloudnet(a)


if __name__ == "__main__":
    main()
