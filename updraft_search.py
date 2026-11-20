#!/usr/bin/env python
"""Find Doppler-lidar vertical-velocity data for the cloud days of every site (ARM and Cloudnet).

ARM sites (ENA, SGP, LASIC, EPCAPE, Bankhead): the vertical-velocity statistics VAP (dlprofwstats4news,
w variance / skewness profiles) and the raw vertical stare (dlfpt), queried with the ARM Live API.
Cloudnet sites: the Cloudnet "doppler-lidar" product (vertical stare, radial velocity v) from the public
Cloudnet API. The Cloudnet site id is matched from a list of aliases (Puijo -> kuopio, Pallas -> pallas /
kenttarova, ...); unmatched sites are reported.

Writes results/updraft_inventory.csv (site, source, dataset, cloud days, days with files, fraction) and
prints a summary. Nothing is downloaded.

Needs network (login node), the observation repo (--obs-repo or $OBS_REPO) and ARM_USER / ARM_TOKEN for ARM.

Examples
--------
python updraft_search.py
python updraft_search.py --no-arm
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from obs_composition_download import cloud_days  # noqa: E402
from sites import SITES  # noqa: E402

CLOUDNET = "https://cloudnet.fmi.fi/api"

# ARM: (members, streams), best first
ARM_DL = {
    "ena":      (["ena", "ena_cloudnet"], ["enadlprofwstats4newsC1.c1", "enadlfptC1.b1"]),
    "sgp":      (["sgp", "sgp_cloudnet"], ["sgpdlprofwstats4newsC1.c1", "sgpdlfptC1.b1",
                                           "sgpdlprofwstats4newsE32.c1", "sgpdlprofwstats4newsE37.c1"]),
    "lasic":    (["lasic", "lasic_cloudnet"], ["asidlprofwstats4newsM1.c1", "asidlfptM1.b1"]),
    "epcape":   (["epcape_pier", "epcape_soledad"], ["epcdlprofwstats4newsM1.c1", "epcdlfptM1.b1",
                                                     "epcdlprofwstats4newsS2.c1", "epcdlfptS2.b1"]),
    "bankhead": (["bankhead"], ["bnfdlprofwstats4newsM1.c1", "bnfdlfptM1.b1"]),
}

# Cloudnet site ids to try, in order (first with doppler-lidar files wins)
CLOUDNET_ALIASES = {
    "puijo": ["kuopio", "puijo"], "pallas": ["pallas", "kenttarova"], "kenttarova": ["kenttarova", "pallas"],
    "zeppelin": ["ny-alesund"], "ny-alesund": ["ny-alesund"], "epcape_soledad": [], "epcape_pier": [],
    "mace-head": ["mace-head"], "hyytiala": ["hyytiala"], "cabauw": ["cabauw"], "granada": ["granada"],
    "palaiseau": ["palaiseau"], "schneefernerhaus": ["schneefernerhaus"], "norunda": ["norunda"],
    "melpitz": ["melpitz", "leipzig"], "bucharest": ["bucharest"], "bankhead": ["bnf", "bankhead"],
    "ena_cloudnet": ["arm-ena", "ena"], "sgp_cloudnet": ["arm-sgp", "sgp"], "lasic_cloudnet": [],
}


def get_json(url, params):
    q = url + "?" + urllib.parse.urlencode(params, doseq=True)
    with urllib.request.urlopen(q, timeout=120) as r:
        return json.loads(r.read().decode())


def cloudnet_part(sites, rows):
    try:
        known = {s["id"] for s in get_json(f"{CLOUDNET}/sites", {})}
    except Exception as e:                                                          # noqa: BLE001
        print(f"Cloudnet: site list failed: {e}")
        return
    for site in sites:
        days = cloud_days(site)
        if not days:
            continue
        cands = [c for c in CLOUDNET_ALIASES.get(site, [site]) if c in known]
        if not cands:
            print(f"  {site:18s} no Cloudnet site id among {CLOUDNET_ALIASES.get(site, [site])}")
            continue
        d0, d1 = min(days).strftime("%Y-%m-%d"), max(days).strftime("%Y-%m-%d")
        best = None
        for cid in cands:
            try:
                files = get_json(f"{CLOUDNET}/files", {"site": cid, "product": "doppler-lidar",
                                                       "dateFrom": d0, "dateTo": d1})
            except Exception as e:                                                  # noqa: BLE001
                print(f"  {site:18s} {cid}: query failed: {str(e)[:80]}")
                continue
            have = {pd.Timestamp(f["measurementDate"]) for f in files}
            n = len(have & days)
            mb = sum(float(f.get("size", 0)) for f in files if pd.Timestamp(f["measurementDate"]) in days) / 1e6
            if best is None or n > best[1]:
                best = (cid, n, mb)
        if best:
            cid, n, mb = best
            print(f"  {site:18s} cloudnet:{cid:12s} {n:4d}/{len(days):4d} cloud days, {mb:8.0f} MB")
            rows.append(dict(site=site, source="Cloudnet", dataset=f"{cid}/doppler-lidar", cloud_days=len(days),
                             days_with_files=n, frac=n / len(days), mb=mb))


def arm_part(rows):
    from arm_ena_nd_pnsd import _file_date, arm_query
    user, token = os.environ.get("ARM_USER", "").strip(), os.environ.get("ARM_TOKEN", "").strip()
    if not user or not token:
        print("ARM: set ARM_USER and ARM_TOKEN; skipping ARM.")
        return
    for group, (members, streams) in ARM_DL.items():
        days = set().union(*(cloud_days(s) for s in members))
        if not days:
            continue
        start = min(days).strftime("%Y-%m-%d")
        end = (max(days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for ds in streams:
            try:
                files = arm_query(user, token, ds, start, end)
            except Exception as e:                                                  # noqa: BLE001
                print(f"  {ds:30s} query error: {str(e)[:80]}")
                continue
            fdays = {_file_date(f).normalize() for f in files if _file_date(f) is not None}
            for s in members:
                sd = cloud_days(s)
                if not sd:
                    continue
                n = len(fdays & sd)
                print(f"  {s:18s} {ds:30s} {n:4d}/{len(sd):4d} cloud days")
                rows.append(dict(site=s, source="ARM", dataset=ds, cloud_days=len(sd), days_with_files=n,
                                 frac=n / len(sd), mb=float("nan")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--no-cloudnet", action="store_true")
    a = ap.parse_args()
    rows = []
    if not a.no_arm:
        if not a.obs_repo:
            ap.error("give --obs-repo or set OBS_REPO (needed for arm_ena_nd_pnsd.py), or --no-arm")
        sys.path.insert(0, a.obs_repo)
        print("ARM Doppler lidar")
        arm_part(rows)
    if not a.no_cloudnet:
        print("\nCloudnet doppler-lidar")
        cloudnet_part([s for s in SITES if s not in ("ena", "sgp", "lasic", "epcape_pier", "epcape_soledad")], rows)
    if not rows:
        return
    t = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "updraft_inventory.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        try:
            old = pd.read_csv(out)
            t = pd.concat([old[~old.source.isin(t.source.unique())], t], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    t.to_csv(out, index=False)
    best = t.sort_values("days_with_files", ascending=False).groupby("site").head(1).sort_values("site")
    print("\nbest source per site:")
    print(best[["site", "source", "dataset", "days_with_files", "cloud_days", "frac"]].to_string(index=False))
    print(f"\n-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
