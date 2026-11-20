#!/usr/bin/env python
"""Which measured aerosol composition exists for the cloud hours of each site?

For every site with an inputs/<site>_parcel_inputs.nc, lists the composition-related datasets
and how many of the site's cloud hours fall inside each one:

  acsm_ams      ACSM / AMS non-refractory PM1 (organics, sulfate, nitrate, ammonium, chloride)
  filter_ions   filter-pack inorganic ions (EMEP 1-/3-pack: sulfate, nitrate, ammonium, sodium, chloride ...)
  oc_ec         filter organic / elemental carbon
  black_carbon  equivalent black carbon / absorption (aethalometer, MAAP, PSAP, SP2)
  ccn           CCN counter (for an independent hygroscopicity check)

EBAS (European sites): the THREDDS catalogue is crawled once and cached in
results/ebas_catalog_names.txt (use --refresh to redo). Stations within --max-km of each site are
matched using station coordinates read over OPeNDAP (cached in results/ebas_station_coords.csv), and
for the candidate files the component list is read from the file's global attributes, so a filter
pack that only holds metals is not counted as ions.

ARM (ENA, SGP, LASIC, EPCAPE, Bankhead): candidate datastreams are queried on ARM Live with
ARM_USER / ARM_TOKEN, and an hour counts as covered when a daily file exists for its date.
Datastream names are guesses where marked; add others with --arm-ds.

Coverage means "inside a dataset's period" (EBAS) or "a file exists that day" (ARM); gaps inside a
period are not checked here.

Needs the observation repo for ebas_psd_search.py and arm_ena_nd_pnsd.py (--obs-repo or $OBS_REPO),
network access to thredds.nilu.no and adc.arm.gov, and requests + netCDF4.

Examples
--------
python composition_search.py                                  # all sites with inputs, EBAS + ARM
python composition_search.py --sites hyytiala puijo pallas --no-arm
python composition_search.py --arm-only --sites ena sgp
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))

# (lat, lon) of the European sites; EBAS stations within --max-km are matched.
EBAS_SITES = {
    "hyytiala": (61.847, 24.295), "mace-head": (53.326, -9.899), "cabauw": (51.970, 4.926),
    "melpitz": (51.530, 12.928), "palaiseau": (48.713, 2.208), "bucharest": (44.348, 26.029),
    "granada": (37.164, -3.605), "schneefernerhaus": (47.417, 10.980), "norunda": (60.087, 17.480),
    "pallas": (67.973, 24.116), "kenttarova": (67.987, 24.243), "zeppelin": (78.907, 11.888),
    "ny-alesund": (78.923, 11.922), "puijo": (62.909, 27.655),
}

# ARM site code and facilities per parcel site.
ARM_SITES = {
    "ena": ("ena", ["C1"]), "ena_cloudnet": ("ena", ["C1"]),
    "sgp": ("sgp", ["E13", "C1"]), "sgp_cloudnet": ("sgp", ["E13", "C1"]),
    "lasic": ("asi", ["M1"]), "lasic_cloudnet": ("asi", ["M1"]),
    "epcape_pier": ("epc", ["M1"]), "epcape_soledad": ("epc", ["M1", "S2"]),
    "bankhead": ("bnf", ["M1"]),
}

# ARM datastream name patterns per category. The ACSM ones are documented; the others are best
# guesses - "no files" for a pattern only means that name does not exist, not that nothing does.
ARM_PATTERNS = {
    "acsm_ams": ["{s}aosacsm{f}.b1", "{s}aosacsm{f}.b2", "{s}aosacsmcdce{f}.c1", "{s}aosacsm{f}.a1"],
    "black_carbon": ["{s}aosaeth2spot{f}.b1", "{s}aosaethsmooth{f}.c1", "{s}aosaeth1f{f}.b1",
                     "{s}aossp2bc60s{f}.b1", "{s}aospsap3w1m{f}.b1"],
    "ccn": ["{s}aosccn1colavg{f}.b1", "{s}aosccn2colaavg{f}.b1", "{s}aosccn1colspectra{f}.b1"],
}

IONS = ("sulphate", "sulfate", "nitrate", "ammonium", "sodium", "chloride", "magnesium", "calcium", "potassium")
AMS_INSTR = ("ams", "hr-tof-ams", "c-tof-ams", "tof-ams", "q-ams")


# ----------------------------------------------------------------------------- helpers
def cloud_hours(site):
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    if not os.path.exists(p):
        return None
    with xr.open_dataset(p) as ds:
        return pd.DatetimeIndex(ds["hour"].values)


def coarse_class(p):
    """Category from the file name alone (cheap first pass); None if clearly not composition."""
    instr, comp, matrix, name = p["instrument"], p["component"], p["matrix"], p["file"].lower()
    if "acsm" in instr or instr in AMS_INSTR or "non_refractory" in matrix:
        return "acsm_ams"
    if instr.startswith("filter") and "absorption" not in instr:
        if any(k in comp for k in ("organic_carbon", "elemental_carbon", "total_carbon")):
            return "oc_ec"
        return "filter_ions?"                       # confirm from the component list
    if any(k in comp for k in ("organic_carbon", "elemental_carbon", "total_carbon")):
        return "oc_ec"
    if ("absorption" in instr or "aethalometer" in instr or "maap" in instr or "psap" in instr
            or "sp2" in instr or "equivalent_black_carbon" in name or "absorption_coefficient" in comp):
        return "black_carbon"
    if "cloud_condensation_nuclei" in comp or instr.startswith("ccn"):
        return "ccn"
    return None


def file_components(fname, thredds):
    """Component keywords from a file's global attributes (OPeNDAP, metadata only)."""
    import netCDF4
    try:
        with netCDF4.Dataset(f"{thredds}/dodsC/ebas/{fname}") as nc:
            at = {k: nc.getncattr(k) for k in nc.ncattrs()}
        txt = " ".join(str(at.get(k, "")) for k in ("keywords", "ebas_component", "summary")).lower()
        return txt
    except Exception as e:                                                   # noqa: BLE001
        return f"!error {type(e).__name__}"


def refine_class(cls, comps):
    if cls != "filter_ions?":
        return cls
    if comps.startswith("!error"):
        return "filter (components unread)"
    if any(i in comps for i in IONS):
        return "filter_ions"
    if any(k in comps for k in ("organic_carbon", "elemental_carbon")):
        return "oc_ec"
    return None                                     # e.g. metals only


def covered(hours, start, end):
    if hours is None:
        return np.nan
    return int(((hours >= pd.Timestamp(start)) & (hours < pd.Timestamp(end))).sum())


# ----------------------------------------------------------------------------- EBAS
def ebas_part(sites, a):
    import requests
    from ebas_psd_search import THREDDS, ebas_catalog, km, parse_ebas, station_coords

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    cache = os.path.join(HERE, "results", "ebas_catalog_names.txt")
    if os.path.exists(cache) and not a.refresh:
        names = open(cache).read().split()
        print(f"EBAS catalogue: {len(names)} file names from cache")
    else:
        print("EBAS catalogue: crawling thredds.nilu.no (several minutes) ...", flush=True)
        names = ebas_catalog(requests.Session())
        open(cache, "w").write("\n".join(names))
        print(f"  {len(names)} file names, cached in {os.path.relpath(cache, HERE)}")

    levels = {x.lower() for x in a.levels}
    cand = []
    for n in names:
        p = parse_ebas(n)
        if p and p["level"] in levels:
            c = coarse_class(p)
            if c:
                p["cls0"] = c
                cand.append(p)
    print(f"  {len(cand)} composition-related files at levels {sorted(levels)}")

    first = {}
    for p in cand:
        first.setdefault(p["station"], p["file"])
    print(f"  reading coordinates of {len(first)} stations (cached after the first run) ...", flush=True)
    coords = station_coords(first, os.path.join(HERE, "results", "ebas_station_coords.csv"))

    rows = []
    for site in sites:
        if site not in EBAS_SITES:
            continue
        lat0, lon0 = EBAS_SITES[site]
        hours = cloud_hours(site)
        near = []
        for p in cand:
            lat, lon, alt, sname = coords.get(p["station"], (np.nan,) * 3 + (p["station"],))
            if lat != lat:
                continue
            d = km(lat0, lon0, lat, lon)
            if d <= a.max_km:
                near.append((p, d, alt, sname))
        print(f"{site}: {len(near)} candidate files within {a.max_km:g} km", flush=True)
        for p, d, alt, sname in near:
            comps = file_components(p["file"], THREDDS) if p["cls0"] == "filter_ions?" else ""
            cls = refine_class(p["cls0"], comps)
            if cls is None:
                continue
            rows.append(dict(site=site, source="EBAS", category=cls, station=p["station"], station_name=sname,
                             km=round(d, 1), instrument=p["instrument"], component=p["component"] or "(multi)",
                             matrix=p["matrix"], resolution=p["resolution"], level=p["level"],
                             start=p["start"].date(), end=p["end"].date(),
                             cloud_hours=len(hours) if hours is not None else np.nan,
                             hours_covered=covered(hours, p["start"], p["end"]), dataset=p["file"]))
    return rows


# ----------------------------------------------------------------------------- ARM
def arm_part(sites, a):
    from arm_ena_nd_pnsd import _file_date, arm_query
    user, token = os.environ.get("ARM_USER"), os.environ.get("ARM_TOKEN")
    if not user or not token:
        print("ARM: set ARM_USER and ARM_TOKEN to query ARM Live; skipping ARM sites.")
        return []
    rows, queried = [], {}
    for site in sites:
        if site not in ARM_SITES:
            continue
        code, facs = ARM_SITES[site]
        hours = cloud_hours(site)
        if hours is None:
            continue
        start, end = (hours.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), \
                     (hours.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hdates = pd.Series(hours.normalize())
        dss = [(cat, pat.format(s=code, f=f)) for cat, pats in ARM_PATTERNS.items() for pat in pats for f in facs]
        dss += [("extra", d) for d in (a.arm_ds or []) if d.startswith(code)]
        print(f"{site}: querying {len(dss)} ARM datastreams {start} .. {end}", flush=True)
        for cat, ds in dss:
            key = (ds, start, end)
            if key not in queried:
                try:
                    queried[key] = arm_query(user.strip(), token.strip(), ds, start, end)
                except Exception as e:                                           # noqa: BLE001
                    print(f"  {ds:30s} query error: {str(e)[:80]}")
                    queried[key] = None
            files = queried[key]
            if not files:
                continue
            days = {pd.Timestamp(d).normalize() for d in map(_file_date, files) if d is not None}
            n = int(hdates.isin(days).sum())
            print(f"  {ds:30s} {len(files):5d} files  covers {n} of {len(hours)} cloud hours")
            rows.append(dict(site=site, source="ARM", category=cat, station=f"{code}{'/'.join(facs)}",
                             station_name=code, km=0.0, instrument=ds.split(".")[0], component="",
                             matrix="", resolution="", level=ds.split(".")[-1],
                             start=min(days).date(), end=max(days).date(), cloud_hours=len(hours),
                             hours_covered=n, dataset=ds))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", help="default: every site with an inputs file")
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"),
                    help="observation repo with ebas_psd_search.py and arm_ena_nd_pnsd.py ($OBS_REPO)")
    ap.add_argument("--max-km", type=float, default=30.0)
    ap.add_argument("--levels", nargs="+", default=["lev2", "lev1.5"])
    ap.add_argument("--refresh", action="store_true", help="re-crawl the EBAS catalogue")
    ap.add_argument("--arm-ds", nargs="*", help="extra ARM datastreams to test")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-arm", action="store_true")
    g.add_argument("--arm-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "composition_inventory.csv"))
    a = ap.parse_args()
    if not a.obs_repo:
        ap.error("give --obs-repo or set OBS_REPO")
    sys.path.insert(0, a.obs_repo)

    sites = a.sites or sorted(f[:-len("_parcel_inputs.nc")] for f in os.listdir(os.path.join(HERE, "inputs"))
                              if f.endswith("_parcel_inputs.nc"))
    rows = []
    if not a.arm_only:
        rows += ebas_part(sites, a)
    if not a.no_arm:
        rows += arm_part(sites, a)
    t = pd.DataFrame(rows)
    if t.empty:
        print(f"nothing found; {os.path.relpath(a.out, HERE)} left unchanged")
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:   # keep rows of the source not searched now
        try:
            old = pd.read_csv(a.out)
            if "source" in old:
                t = pd.concat([old[~old.source.isin(t.source.unique())], t], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    t.to_csv(a.out, index=False)

    # per site and category: best single dataset and the union of dataset periods
    summ = []
    for (site, cat), g in t.groupby(["site", "category"]):
        hours = cloud_hours(site)
        if hours is not None and g.source.iloc[0] == "EBAS":
            m = np.zeros(len(hours), bool)
            for s, e in zip(g.start, g.end):
                m |= (hours >= pd.Timestamp(s)) & (hours < pd.Timestamp(e))
            union = int(m.sum())
        else:
            union = int(g.hours_covered.max())
        summ.append(dict(site=site, category=cat, cloud_hours=g.cloud_hours.iloc[0], hours_covered=union,
                         datasets=len(g), years=f"{min(g.start)}..{max(g.end)}",
                         instruments=",".join(sorted(set(g.instrument)))[:60]))
    s = pd.DataFrame(summ)
    wide = s.pivot(index="site", columns="category", values="hours_covered")
    wide.insert(0, "cloud_hours", s.groupby("site").cloud_hours.first())
    with pd.option_context("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 500):
        print("\nCloud hours inside a dataset period, by site and category:\n")
        print(wide.fillna(0).astype(int).to_string())
        print("\nDetails per site and category:\n")
        print(s.to_string(index=False))
    s.to_csv(a.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nwrote {os.path.relpath(a.out, HERE)} and {os.path.relpath(a.out.replace('.csv', '_summary.csv'), HERE)} "
          f"({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")


if __name__ == "__main__":
    main()
