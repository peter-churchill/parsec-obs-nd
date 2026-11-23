#!/usr/bin/env python
"""Download surface meteorology (temperature, relative humidity, pressure) for the cloud days of every site.

Used by build_thermo.py to get cloud-base T and p from the surface: dry adiabat from the surface to the
measured cloud base where the sub-cloud layer is coupled (cloud base near the LCL), and the station itself
at the in-cloud stations.

  ARM sites           MET b1 streams (1-min temp_mean, rh_mean, atmos_pressure) -> data/met/arm/<stream>/
  Cloudnet sites      Cloudnet weather-station product (air_temperature, relative_humidity, air_pressure)
                      -> data/met/cloudnet/<site id>/
  Puijo, Pallas,      FMI open data, hourly TA / RH / PA (Kuopio Puijo tower, Sammaltunturi, Kenttarova);
  Kenttarova          stations found by name in FMI's station list -> data/met/fmi/<site>.csv

--dry-run only reports what exists for the cloud days. Files already present are skipped. Needs network,
OBS_REPO and ARM_USER / ARM_TOKEN for ARM.

Examples
--------
python met_download.py --dry-run
nohup python -u met_download.py > logs/met_download.log 2>&1 &
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
from updraft_download import fetch, get_json  # noqa: E402
from updraft_search import CLOUDNET, CLOUDNET_ALIASES  # noqa: E402
from sites import SITES  # noqa: E402

OUT = os.path.join(HERE, "data", "met")
ARM_MET = {
    "ena": ["enametC1.b1"], "ena_cloudnet": ["enametC1.b1"],
    "sgp": ["sgpmetE13.b1"], "sgp_cloudnet": ["sgpmetE13.b1"],
    "lasic": ["asimetM1.b1"], "lasic_cloudnet": ["asimetM1.b1"],
    "epcape_pier": ["epcmetM1.b1"], "epcape_soledad": ["epcmetS2.b1", "epcmetM1.b1"],
    "bankhead": ["bnfmetM1.b1"],
}
# FMI stations, matched by name in FMI's station list (fmi::ef::stations) and then queried by fmisid
# station name parts, tried in order until one station returns data (the Puijo tower station has no open data
# for Puijo's cloud years, so Puijo falls back to Kuopio Savilahti, a few km away at lake level)
FMI_STATIONS = {"puijo": ["Puijo", "Savilahti"], "pallas": ["Sammaltunturi"], "kenttarova": ["Kenttärova"]}
# altitude (m a.s.l.) of FMI stations that are not at the site's own altitude; build_thermo.py carries their
# T, RH, p up to the site (dry adiabat to the LCL, moist above it)
FMI_ALT = {"Kuopio Savilahti": 85.0}
FMI_WFS = "https://opendata.fmi.fi/wfs"


def arm(a):
    from arm_ena_nd_pnsd import _file_date, arm_download, arm_query
    user, token = os.environ.get("ARM_USER", "").strip(), os.environ.get("ARM_TOKEN", "").strip()
    if not user or not token:
        print("ARM: set ARM_USER and ARM_TOKEN; skipping")
        return
    want = {}
    for site, streams in ARM_MET.items():
        if a.sites and site not in a.sites:
            continue
        for s in streams:
            want.setdefault(s, set()).update(cloud_days(site))
    for stream, days in want.items():
        if not days:
            continue
        try:
            files = arm_query(user, token, stream, min(days).strftime("%Y-%m-%d"),
                              (max(days) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        except Exception as e:                                                      # noqa: BLE001
            print(f"ARM {stream:16s} query error: {str(e)[:80]}")
            continue
        files = [f for f in files if _file_date(f) is not None and _file_date(f).normalize() in days]
        print(f"ARM {stream:16s} {len(files):4d} files for {len(days)} cloud days", flush=True)
        if a.dry_run or not files:
            continue
        outdir = os.path.join(OUT, "arm", stream)
        os.makedirs(outdir, exist_ok=True)
        res = [arm_download(user, token, f, outdir) for f in files]
        print(f"    downloaded {res.count('downloaded')}, failed {res.count('failed')}", flush=True)


def cloudnet(a):
    known = {s["id"] for s in get_json(f"{CLOUDNET}/sites", {})}
    for site, cfg in SITES.items():
        if cfg["kind"] != "cloudnet" or (a.sites and site not in a.sites):
            continue
        days = cloud_days(site)
        if not days:
            continue
        best = None
        for cid in [c for c in CLOUDNET_ALIASES.get(site, [site]) if c in known]:
            files = get_json(f"{CLOUDNET}/files", {"site": cid, "product": "weather-station",
                                                   "dateFrom": min(days).strftime("%Y-%m-%d"),
                                                   "dateTo": max(days).strftime("%Y-%m-%d")})
            files = [f for f in files if pd.Timestamp(f["measurementDate"]) in days]
            if best is None or len(files) > len(best[1]):
                best = (cid, files)
        if best is None:
            print(f"Cloudnet {site:16s} no Cloudnet site id")
            continue
        cid, files = best
        print(f"Cloudnet {site:16s} ({cid:16s}) weather-station {len(files):4d}/{len(days):4d} cloud days", flush=True)
        if a.dry_run or not files:
            continue
        outdir = os.path.join(OUT, "cloudnet", cid)
        os.makedirs(outdir, exist_ok=True)
        res = [fetch(f["downloadUrl"], os.path.join(outdir, f["filename"])) for f in files]
        print(f"    downloaded {res.count('downloaded')}, skipped {res.count('skipped')}, failed {res.count('failed')}")


def fmi_stations():
    """{fmisid: [names]} from FMI's station list."""
    import xml.etree.ElementTree as ET
    q = {"service": "WFS", "version": "2.0.0", "request": "getFeature", "storedquery_id": "fmi::ef::stations"}
    with urllib.request.urlopen(FMI_WFS + "?" + urllib.parse.urlencode(q), timeout=180) as r:
        root = ET.fromstring(r.read())
    gml = "{http://www.opengis.net/gml/3.2}"
    out = {}
    for fac in (e for e in root.iter() if str(e.tag).endswith("}EnvironmentalMonitoringFacility")):
        ident = fac.find(f"{gml}identifier")
        if ident is not None:
            out[ident.text.strip()] = [n.text for n in fac.findall(f"{gml}name") if n.text]
    return out


def fmi_month(fmisid, start, end):
    import xml.etree.ElementTree as ET
    q = {"service": "WFS", "version": "2.0.0", "request": "getFeature",
         "storedquery_id": "fmi::observations::weather::hourly::simple", "fmisid": fmisid,
         "parameters": "TA_PT1H_AVG,RH_PT1H_AVG,PA_PT1H_AVG",
         "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "endtime": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    with urllib.request.urlopen(FMI_WFS + "?" + urllib.parse.urlencode(q), timeout=120) as r:
        root = ET.fromstring(r.read())
    b = "{http://xml.fmi.fi/schema/wfs/2.0}"
    rows = [dict(time=el.find(f"{b}Time").text, param=el.find(f"{b}ParameterName").text,
                 value=float(el.find(f"{b}ParameterValue").text)) for el in root.iter(f"{b}BsWfsElement")]
    return pd.DataFrame(rows)


def fmi_10min(fmisid, start, end):
    """10-min observations (t2m, rh, p_sea) in 7-day chunks, for stations without hourly averages."""
    import xml.etree.ElementTree as ET
    b = "{http://xml.fmi.fi/schema/wfs/2.0}"
    rows = []
    t = start
    while t < end:
        t1 = min(t + pd.Timedelta(days=7), end)
        q = {"service": "WFS", "version": "2.0.0", "request": "getFeature",
             "storedquery_id": "fmi::observations::weather::simple", "fmisid": fmisid,
             "parameters": "t2m,rh,p_sea", "timestep": "10",
             "starttime": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "endtime": t1.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with urllib.request.urlopen(FMI_WFS + "?" + urllib.parse.urlencode(q), timeout=120) as r:
            root = ET.fromstring(r.read())
        rows += [dict(time=el.find(f"{b}Time").text, param=el.find(f"{b}ParameterName").text,
                      value=float(el.find(f"{b}ParameterValue").text)) for el in root.iter(f"{b}BsWfsElement")]
        t = t1
    return pd.DataFrame(rows)


def fmi(a):
    try:
        stations = fmi_stations()
    except Exception as e:                                                          # noqa: BLE001
        print(f"FMI: station list failed: {str(e)[:100]}")
        return
    for site, keys in FMI_STATIONS.items():
        if a.sites and site not in a.sites:
            continue
        days = cloud_days(site)
        if not days:
            continue
        cands = [(fid, names) for k in keys for fid, names in stations.items()
                 if k.lower() in " ".join(names).lower()]
        if not cands:
            print(f"FMI {site}: no station matching {keys}; ERA5 will be used")
            continue
        out = os.path.join(OUT, "fmi", f"{site}.csv")
        months = sorted({pd.Period(d, "M") for d in days})
        print(f"FMI {site:10s} {len(months)} months with cloud days; stations in order: "
              + "; ".join(f"{f} '{n[0]}'" for f, n in cands), flush=True)
        if a.dry_run or os.path.exists(out):
            continue
        for fid, names in cands:
            parts = fetch_station(fid, months)
            if parts:
                df = pd.concat(parts).pivot_table(index="time", columns="param", values="value")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                df.to_csv(out)
                with open(out.replace(".csv", "_station.json"), "w") as fh:
                    json.dump({"fmisid": fid, "name": names[0], "altitude_m": FMI_ALT.get(names[0])}, fh)
                print(f"    -> {os.path.relpath(out, HERE)}: {len(df)} hours with data from {fid} '{names[0]}'")
                break
            print(f"    {fid} '{names[0]}': no data for these months")
        else:
            print(f"    FMI returned no data for {site}; ERA5 / Cloudnet model will be used")


def fetch_station(fid, months):
    """Hourly averages for the months, else 10-min observations averaged to hours (p_sea as P_SEA)."""
    parts = []
    for m in months:
        try:
            parts.append(fmi_month(fid, m.start_time, m.end_time.floor("h")))
        except Exception as e:                                                      # noqa: BLE001
            print(f"    {m}: {str(e)[:80]}")
    parts = [p for p in parts if len(p) and p["value"].notna().any()]
    if parts:
        return parts
    print(f"    {fid}: no hourly averages; trying 10-min observations", flush=True)
    for m in months:
        try:
            d = fmi_10min(fid, m.start_time, m.end_time.floor("h"))
            if len(d):
                d["time"] = (pd.to_datetime(d["time"]).dt.ceil("h")).astype(str)
                d = d.groupby(["time", "param"], as_index=False)["value"].mean()
                d["param"] = d["param"].map({"t2m": "TA_PT1H_AVG", "rh": "RH_PT1H_AVG", "p_sea": "P_SEA"})
                parts.append(d)
        except Exception as e:                                                      # noqa: BLE001
            print(f"    {m}: {str(e)[:80]}")
    return [p for p in parts if len(p) and p["value"].notna().any()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--source", choices=["arm", "cloudnet", "fmi", "all"], default="all")
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.source in ("arm", "all"):
        if not a.obs_repo:
            ap.error("give --obs-repo or set OBS_REPO for ARM, or --source cloudnet / fmi")
        sys.path.insert(0, a.obs_repo)
        arm(a)
    if a.source in ("cloudnet", "all"):
        cloudnet(a)
    if a.source in ("fmi", "all"):
        fmi(a)


if __name__ == "__main__":
    main()
