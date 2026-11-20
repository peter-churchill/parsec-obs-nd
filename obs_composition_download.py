#!/usr/bin/env python
"""Download the measured aerosol composition (and CCN) for the cloud hours of each site.

ARM sites: the ACSM, PSAP / SP2 (black carbon) and CCN datastreams found by composition_search.py,
only for the days with cloud hours. Files go to data/obs_comp/arm/<group>/<datastream>/.

EBAS sites: every dataset in results/composition_inventory.csv (written by composition_search.py)
that covers at least one cloud hour, in the categories asked for. Files go to
data/obs_comp/ebas/<station>/ and are shared by sites on the same station (Pallas and Kenttarova,
Zeppelin and Ny-Alesund).

A manifest, data/obs_comp/manifest.csv, lists which files belong to which site, category and source;
the composition step reads it. Files already present are skipped, so the script can be rerun.

Needs the observation repo (--obs-repo or $OBS_REPO) for arm_ena_nd_pnsd.py, ARM_USER / ARM_TOKEN
for ARM, and network access to adc.arm.gov and thredds.nilu.no (login node).

Examples
--------
python obs_composition_download.py --dry-run
python obs_composition_download.py --source ebas
nohup python -u obs_composition_download.py > logs/obs_comp_download.log 2>&1 &
"""
import argparse
import os
import sys
import time

import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "obs_comp")
THREDDS = "https://thredds.nilu.no/thredds"

# ARM groups: member parcel sites and the datastreams per category, best first. Names are those that
# composition_search.py found files for (Bankhead's are assumed by analogy; missing ones are skipped).
ARM_GROUPS = {
    "ena":    (["ena", "ena_cloudnet"],
               {"acsm": ["enaaosacsmC1.b2", "enaaosacsmC1.b1"],
                "black_carbon": ["enaaospsap3w1mC1.b1"],
                "ccn": ["enaaosccn2colaavgC1.b1"]}),
    "sgp":    (["sgp", "sgp_cloudnet"],
               {"acsm": ["sgpaosacsmE13.b2", "sgpaosacsmE13.b1", "sgpaosacsmC1.b1"],
                "black_carbon": ["sgpaospsap3w1mE13.b1", "sgpaospsap3w1mC1.b1"],
                "ccn": ["sgpaosccn2colaavgE13.b1", "sgpaosccn1colavgC1.b1"]}),
    "lasic":  (["lasic", "lasic_cloudnet"],
               {"acsm": ["asiaosacsmM1.b2", "asiaosacsmM1.b1"],
                "black_carbon": ["asiaospsap3w1mM1.b1"],
                "ccn": ["asiaosccn2colaavgM1.b1"]}),
    "epcape": (["epcape_pier", "epcape_soledad"],
               {"acsm": ["epcaosacsmM1.b2", "epcaosacsmM1.b1"],
                "black_carbon": ["epcaospsap3w1mM1.b1", "epcaossp2bc60sM1.b1"],
                "ccn": ["epcaosccn2colaavgM1.b1"]}),
    "bankhead": (["bankhead"],
               {"acsm": ["bnfaosacsmM1.b2", "bnfaosacsmM1.b1"],
                "black_carbon": ["bnfaospsap3w1mM1.b1"],
                "ccn": ["bnfaosccn2colaavgM1.b1"]}),
}
EBAS_CATEGORIES = ["acsm_ams", "filter_ions", "black_carbon", "ccn"]


def cloud_days(site):
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(p):
        return set()
    with xr.open_dataset(p) as ds:
        return set(pd.DatetimeIndex(ds["hour"].values).normalize())


# ----------------------------------------------------------------------------- ARM
def arm_part(a, manifest):
    from arm_ena_nd_pnsd import _file_date, arm_download, arm_query
    user, token = os.environ.get("ARM_USER", "").strip(), os.environ.get("ARM_TOKEN", "").strip()
    if not user or not token:
        print("ARM: set ARM_USER and ARM_TOKEN (your real values); skipping ARM.")
        return
    for group, (members, streams) in ARM_GROUPS.items():
        days = set().union(*(cloud_days(s) for s in members))
        if not days:
            continue
        start = (min(days) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (max(days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"\nARM {group}: {len(days)} cloud days, {start} .. {end}", flush=True)
        for cat, dss in streams.items():
            if cat not in a.categories_arm:
                continue
            for ds in dss:
                try:
                    files = arm_query(user, token, ds, start, end)
                except Exception as e:                                               # noqa: BLE001
                    print(f"  {ds:28s} query error: {str(e)[:100]}")
                    continue
                want = [f for f in files if _file_date(f) is not None and _file_date(f).normalize() in days]
                print(f"  {ds:28s} {len(files):5d} files in range, {len(want):5d} on cloud days", flush=True)
                if not want:
                    continue
                outdir = os.path.join(OUT, "arm", group, ds)
                n_new = n_fail = 0
                if not a.dry_run:
                    os.makedirs(outdir, exist_ok=True)
                    for k, f in enumerate(want, 1):
                        r = arm_download(user, token, f, outdir)
                        n_new += r == "downloaded"
                        n_fail += r == "failed"
                        if k % 100 == 0:
                            print(f"    {k}/{len(want)}", flush=True)
                    print(f"    {n_new} downloaded, {n_fail} failed, {len(want) - n_new - n_fail} already there")
                for s in members:
                    manifest.append(dict(site=s, source="ARM", category=cat, dataset=ds,
                                         path=os.path.relpath(outdir, HERE), files=len(want)))


# ----------------------------------------------------------------------------- EBAS
def fetch(url, dest, retries=4):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            with open(tmp, "rb") as f:
                head = f.read(4)
            if not (head[:3] == b"CDF" or head == b"\x89HDF"):
                os.remove(tmp)
                return "not netcdf"
            os.replace(tmp, dest)
            return "downloaded"
        except Exception as e:                                                       # noqa: BLE001
            if attempt == retries:
                print(f"    failed {os.path.basename(dest)}: {str(e)[:100]}")
                return "failed"
            time.sleep(10 * attempt)


def ebas_part(a, manifest):
    inv = os.path.join(HERE, "results", "composition_inventory.csv")
    if not os.path.exists(inv):
        print("EBAS: results/composition_inventory.csv missing; run composition_search.py --no-arm first.")
        return
    t = pd.read_csv(inv)
    t = t[(t.source == "EBAS") & (t.hours_covered > 0) & t.category.isin(a.categories_ebas)]
    files = t.drop_duplicates("dataset")
    print(f"\nEBAS: {len(t)} site-dataset rows, {len(files)} distinct files", flush=True)
    stats = {}
    for _, r in files.iterrows():
        outdir = os.path.join(OUT, "ebas", r.station)
        dest = os.path.join(outdir, r.dataset)
        if a.dry_run:
            res = "would download" if not os.path.exists(dest) else "skipped"
        else:
            os.makedirs(outdir, exist_ok=True)
            res = fetch(f"{THREDDS}/fileServer/ebas/{r.dataset}", dest)
        stats[res] = stats.get(res, 0) + 1
    print("  " + ", ".join(f"{k}: {v}" for k, v in stats.items()))
    for _, r in t.iterrows():
        manifest.append(dict(site=r.site, source="EBAS", category=r.category, dataset=r.dataset,
                             path=os.path.relpath(os.path.join(OUT, "ebas", r.station, r.dataset), HERE),
                             files=1, start=r.start, end=r.end, hours_covered=r.hours_covered,
                             instrument=r.instrument, resolution=r.resolution))
    for cat, g in t.groupby("category"):
        print(f"  {cat:13s} {g.site.nunique():2d} sites, {g.dataset.nunique():3d} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["all", "arm", "ebas"], default="all")
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--categories-arm", nargs="+", default=["acsm", "black_carbon", "ccn"])
    ap.add_argument("--categories-ebas", nargs="+", default=EBAS_CATEGORIES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    manifest = []
    if a.source in ("all", "arm"):
        if not a.obs_repo:
            ap.error("give --obs-repo or set OBS_REPO (needed for arm_ena_nd_pnsd.py)")
        sys.path.insert(0, a.obs_repo)
        arm_part(a, manifest)
    if a.source in ("all", "ebas"):
        ebas_part(a, manifest)
    if manifest and not a.dry_run:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, "manifest.csv")
        m = pd.DataFrame(manifest)
        if os.path.exists(path):                       # keep rows of the source not rerun now
            old = pd.read_csv(path)
            m = pd.concat([old[~old.source.isin(m.source.unique())], m], ignore_index=True)
        m.to_csv(path, index=False)
        print(f"\nmanifest: {os.path.relpath(path, HERE)} ({len(m)} rows)")


if __name__ == "__main__":
    main()
