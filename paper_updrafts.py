#!/usr/bin/env python
"""Puijo and Pallas updrafts from the sources in the paper's data-availability statement.

Paper (Virtanen et al., "High sensitivity of cloud formation to aerosol changes"):
  * Zenodo 10.5281/zenodo.13358529: Puijo DMPS + met + FORCES 2020 updraft, Pallas DMPS total inlet + met,
    Zeppelin anemometer (1-h averages).
  * Pallas Doppler lidar via ACTRIS/Cloudnet: site "kenttarova", product doppler-lidar (FMI HALO), e.g.
    https://cloudnet.fmi.fi/file/06fb58cc-13fc-481d-b695-287901e338ac (2023-12-10).

Neither updraft record overlaps our hours (Puijo 2009-2015, Pallas 2005-2015; FORCES is 2020 and Cloudnet has
Kenttarova lidar only from 2017 on), so, as in the paper, they give a per-station updraft climatology for
in-cloud hours (station visibility below --vis-max), not hour-by-hour values. If any hours do overlap they
are used directly.

Steps
-----
python paper_updrafts.py download                # Zenodo files -> data/zenodo_13358529/ (md5 checked)
python paper_updrafts.py inspect                 # header, first rows, time range of every CSV (paste this)
python paper_updrafts.py pallas-lidar --dry-run  # Kenttarova lidar days with Pallas in cloud
python paper_updrafts.py pallas-lidar            # download them -> data/updraft/cloudnet/kenttarova/
python paper_updrafts.py build                   # updrafts/{puijo,pallas}_w_cloudbase.nc

build writes the same variables as build_updrafts.py (sigma_w, w_mean = 0, hour_source, ...), with sigma_w
the median of the station's in-cloud hourly sigma_w and the 10/90 % values in the attributes and the log, to
use as --sigma-w bounds in weight_updrafts.py / compare_nd.py. Run it after build_updrafts.py (which would
otherwise overwrite these two files with the pooled fallback).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ZEN_ID = "13358529"
ZEN = os.path.join(HERE, "data", f"zenodo_{ZEN_ID}")
LIDAR = os.path.join(HERE, "data", "updraft", "cloudnet", "kenttarova")
CLOUDNET = "https://cloudnet.fmi.fi/api"
# Pallas (Sammaltunturi, 565 m) sits ~220 m above the Kenttarova lidar (345 m): when Pallas is in cloud the
# base is at or below that height, so the lidar layer is taken just below it.
PALLAS_DZ = 565.0 - 345.0


# ----------------------------------------------------------------------------- helpers
def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())


def fetch(url, dest, md5=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and (md5 is None or file_md5(dest) == md5):
        return "skipped"
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    if md5 and file_md5(tmp) != md5:
        os.remove(tmp)
        return "md5 mismatch"
    os.replace(tmp, dest)
    return "downloaded"


def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(name):
    """Zenodo CSV with a parsed hourly 'time' index; delimiter and time format detected."""
    p = os.path.join(ZEN, name)
    df = pd.read_csv(p, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    tcol = next((c for c in df.columns if any(k in c.lower() for k in ("time", "date"))), df.columns[0])
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
        v = t.astype(float)
        if v.median() > 7e5:                                   # MATLAB datenum
            t = pd.to_datetime(v - 719529, unit="D")
        elif v.median() > 1e8:                                 # unix seconds
            t = pd.to_datetime(v, unit="s")
        else:
            raise ValueError(f"{name}: cannot interpret numeric time column {tcol!r}")
    else:
        t = pd.to_datetime(t, errors="coerce", dayfirst=False)
    df.index = pd.DatetimeIndex(t).round("min").floor("h")   # datenum float error
    df = df.drop(columns=[tcol])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.where(df > -9000)                                  # common fill values (-9999 etc.)
    return df[~df.index.isna()]


def col(df, *keys, exclude=()):
    """First column whose lower-case name contains all keys and none of exclude."""
    for c in df.columns:
        lc = c.lower()
        if all(k in lc for k in keys) and not any(e in lc for e in exclude):
            return c
    return None


def cloud_hours(site, vis_max):
    """Hours with station visibility < vis_max (m) and no rain, from the Zenodo met file."""
    m = read_csv(f"{site.capitalize()}_meteorological_data_1h_ave.csv")
    cv, cr = col(m, "vis"), col(m, "rain")
    if cv is None:
        raise SystemExit(f"{site}: no visibility column in met file: {list(m.columns)}")
    ok = m[cv] < vis_max
    if cr is not None:
        ok &= ~(m[cr] > 0)
    return pd.DatetimeIndex(m.index[ok.fillna(False).values]).unique(), m


def inputs_hours(site):
    with xr.open_dataset(os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")) as ds:
        return pd.DatetimeIndex(ds["hour"].values)


# ----------------------------------------------------------------------------- steps
def download(a):
    rec = get_json(f"https://zenodo.org/api/records/{ZEN_ID}")
    os.makedirs(ZEN, exist_ok=True)
    for f in rec["files"]:
        name = f["key"]
        md5 = f.get("checksum", "").replace("md5:", "") or None
        url = f["links"]["self"]
        r = "dry-run" if a.dry_run else fetch(url, os.path.join(ZEN, name), md5)
        print(f"{name:50s} {int(f['size']) / 1e6:7.1f} MB  {r}", flush=True)


def inspect(a):
    for p in sorted(glob.glob(os.path.join(ZEN, "*.csv"))):
        name = os.path.basename(p)
        print(f"\n=== {name}")
        with open(p, errors="replace") as fh:
            for k, line in enumerate(fh):
                if k < 4:
                    print("   " + line.rstrip()[:200])
        if "Dp_nm" in name:
            continue
        try:
            df = read_csv(name)
            print(f"   parsed: {len(df)} rows, {df.index.min()} .. {df.index.max()}, "
                  f"{df.shape[1]} columns: {list(df.columns)[:8]}{' ...' if df.shape[1] > 8 else ''}")
            print("   years: " + ", ".join(f"{y}:{n}" for y, n in df.index.year.value_counts().sort_index().items()))
            small = df.iloc[:, :8]
            print("   " + small.describe(percentiles=[.1, .5, .9]).T.round(3).to_string().replace("\n", "\n   "))
        except Exception as ex:                                                      # noqa: BLE001
            print(f"   could not parse: {ex}")


def pallas_lidar(a):
    hrs, _ = cloud_hours("pallas", a.vis_max)
    days = pd.DatetimeIndex(hrs.normalize().unique())
    files = get_json(f"{CLOUDNET}/files?site=kenttarova&product=doppler-lidar&dateFrom=2017-01-01"
                     f"&dateTo={pd.Timestamp.now():%Y-%m-%d}")
    files = [f for f in files if pd.Timestamp(f["measurementDate"]) in days]
    print(f"Pallas in-cloud hours (vis < {a.vis_max:.0f} m): {len(hrs)} on {len(days)} days "
          f"({days.min():%Y-%m-%d} .. {days.max():%Y-%m-%d})")
    print(f"Kenttarova doppler-lidar files on those days: {len(files)}, "
          f"{sum(float(f['size']) for f in files) / 1e9:.1f} GB")
    if a.dry_run or not files:
        return
    os.makedirs(LIDAR, exist_ok=True)
    res = [fetch(f["downloadUrl"], os.path.join(LIDAR, f["filename"])) for f in files]
    print(f"downloaded {res.count('downloaded')}, skipped {res.count('skipped')}, "
          f"failed {len(res) - res.count('downloaded') - res.count('skipped')}")


def lidar_sigma(hours, z0, z1):
    """Hourly std of the layer-mean vertical velocity between z0 and z1 m above the lidar (>= 30 profiles)."""
    rows = []
    days = set(pd.DatetimeIndex(hours).normalize())
    for f in sorted(glob.glob(os.path.join(LIDAR, "*.nc"))):
        if pd.Timestamp(os.path.basename(f)[:8]) not in days:
            continue
        try:
            with xr.open_dataset(f) as ds:
                t = pd.DatetimeIndex(ds["time"].values)
                hr = t.floor("h")
                keep = hr.isin(hours)
                if not keep.any():
                    continue
                alt = float(np.nanmedian(ds["altitude"].values)) if "altitude" in ds else 345.0
                z = ds["height"].values.astype(float) - alt if "height" in ds else ds["range"].values.astype(float)
                m = (z >= z0) & (z <= z1)
                with np.errstate(all="ignore"):
                    w = np.nanmean(ds["v"].values[keep][:, m], axis=1)
                rows.append(pd.DataFrame(dict(hour=hr[keep], w=w)))
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
    if not rows:
        return pd.Series(dtype=float)
    g = pd.concat(rows).dropna().groupby("hour")["w"]
    s = g.std()
    return s[g.size() >= 30]


def write(site, hours, sigma_hourly, clim, src, a):
    """sigma_w per hour: matched value where available, else the climatological median."""
    q10, q50, q90 = np.percentile(clim, [10, 50, 90])
    sw = pd.Series(q50, index=hours)
    hsrc = pd.Series("paper_climatology", index=hours, dtype=object)
    matched = sigma_hourly.reindex(hours).dropna() if sigma_hourly is not None else pd.Series(dtype=float)
    sw[matched.index] = matched.values
    hsrc[matched.index] = "paper_hourly"
    ds = xr.Dataset({
        "sigma_w": ("hour", sw.values, {"units": "m s-1", "long_name": "w std near cloud base"}),
        "w_mean": ("hour", np.zeros(len(hours)), {"units": "m s-1", "long_name": "mean w used for the PDF"}),
        "w_mean_obs": ("hour", np.full(len(hours), np.nan), {"units": "m s-1"}),
        "n_samples": ("hour", np.full(len(hours), len(clim), dtype=int)),
        "sigma_w_cb": ("hour", np.full(len(hours), np.nan), {"units": "m s-1"}),
        "hour_source": ("hour", hsrc.values)},
        coords={"hour": hours.values},
        attrs={"w_source": src, "sigma_w_p10": q10, "sigma_w_p50": q50, "sigma_w_p90": q90,
               "n_climatology_hours": len(clim), "vis_max_m": a.vis_max})
    os.makedirs(os.path.join(HERE, "updrafts"), exist_ok=True)
    ds.to_netcdf(os.path.join(HERE, "updrafts", f"{site}_w_cloudbase.nc"))
    print(f"{site:8s} {src}: {len(clim)} in-cloud hours, sigma_w 10/50/90 % = {q10:.2f}/{q50:.2f}/{q90:.2f}; "
          f"{len(matched)} of {len(hours)} hours matched directly "
          f"(bounds for compare_nd.py: --sigma-w {q10:.2f} / {q90:.2f})")


def build(a):
    sites = a.sites or ["puijo", "pallas"]
    if "puijo" in sites:
        u = read_csv("Puijo_Forces2020_updraft_temp_data_1h_ave.csv")
        cs = col(u, "sigma") or col(u, "std")
        if cs is None:
            raise SystemExit(f"Puijo updraft file: no sigma column in {list(u.columns)}")
        cr = col(u, "r2") or col(u, "r^2") or col(u, "r_2")
        ok = u[cs].notna() & (u[cs] > 0)
        if cr is not None and a.min_r2 > 0:
            ok &= u[cr] >= a.min_r2
        sig = u.loc[ok, cs]
        try:
            ch, _ = cloud_hours("puijo", a.vis_max)
            inc = sig[sig.index.isin(ch)]
        except Exception as ex:                                                      # noqa: BLE001
            print(f"puijo: no visibility for the FORCES period ({str(ex)[:60]}); using all hours")
            inc = sig
        if len(inc) < 30:
            print(f"puijo: only {len(inc)} in-cloud FORCES hours (met may not cover 2020); using all {len(sig)} hours")
            inc = sig
        sig = sig.groupby(level=0).mean()
        write("puijo", inputs_hours("puijo"), sig, inc.values,
              "zenodo_13358529_Puijo_Forces2020" + ("_incloud" if len(inc) < len(sig) else "_allhours"), a)
    if "pallas" in sites:
        ch, _ = cloud_hours("pallas", a.vis_max)
        s = lidar_sigma(ch, PALLAS_DZ - a.depth, PALLAS_DZ - a.top)
        if len(s) < 30:
            raise SystemExit(f"pallas: only {len(s)} lidar hours with Pallas in cloud; run pallas-lidar first")
        write("pallas", inputs_hours("pallas"), s, s.values,
              f"cloudnet_kenttarova_doppler_lidar_{PALLAS_DZ - a.depth:.0f}-{PALLAS_DZ - a.top:.0f}m_pallas_incloud", a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "inspect", "pallas-lidar", "build"])
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vis-max", type=float, default=200.0, help="in-cloud: station visibility below this (m)")
    ap.add_argument("--min-r2", type=float, default=0.0, help="Puijo FORCES: minimum r2 column value")
    ap.add_argument("--top", type=float, default=0.0, help="Pallas lidar layer top, m below the station height")
    ap.add_argument("--depth", type=float, default=150.0, help="Pallas lidar layer bottom, m below the station")
    a = ap.parse_args()
    {"download": download, "inspect": inspect, "pallas-lidar": pallas_lidar, "build": build}[a.step](a)


if __name__ == "__main__":
    main()
