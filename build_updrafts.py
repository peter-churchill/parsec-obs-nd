#!/usr/bin/env python
"""Hourly updraft statistics just below cloud base from Doppler lidar, for weight_updrafts.py.

For each hour of inputs/<site>_parcel_inputs.nc, vertical velocity is taken in a layer below cloud base,
from CBH - --top m down to CBH - --depth m (default 50-300 m below base), with CBH the hour's cloud base
height from the inputs (above ground), else the lidar's own cloud base.

ARM (dlprofwstats4news, 10-min statistics): per 10-min interval, the layer mean of the noise-corrected
  w_variance and of the median w, at gates with SNR above the file's snr_threshold. Per hour:
  sigma_w = sqrt(mean 10-min variance + variance of the 10-min medians), w_mean = mean of the medians.
  The VAP's own cloud-base velocity quartiles (cbw_25 / cbw_75) give sigma_w_cb = IQR / 1.349 as a check.
Cloudnet (doppler-lidar, vertical stare, v): per profile the layer-mean v (gates the processing kept);
  per hour sigma_w and w_mean over the profiles, and up to --max-samples profiles as w(hour, sample).

Cloudnet radar (categorize; sites without lidar): Doppler velocity in non-drizzling cloud-droplet gates
  from cloud base to --radar-depth m above it, see radar_site().
Fallback (Puijo, Pallas: neither): one sigma_w for all hours, the median of the boreal lidar sites.

Writes updrafts/<site>_w_cloudbase.nc (sigma_w, w_mean, n, sigma_w_cb; w samples for Cloudnet lidar) with
w_source. Zeppelin's file comes from zeppelin_bolin.py and is left alone; the Ny-Alesund radar on
Zeppelin's days goes to updrafts/zeppelin_w_radar.nc for comparison.

Examples
--------
python build_updrafts.py
python build_updrafts.py --sites ena hyytiala --depth 400
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sites import SITE_COORDS  # noqa: E402
from updraft_download import ARM_STREAMS, CLOUDNET_RADAR_SITES, CLOUDNET_SITES  # noqa: E402

# sites with neither lidar nor radar on their cloud days: per-regime sigma_w pooled from these lidar sites
FALLBACK = {"puijo": ["hyytiala", "kenttarova"], "pallas": ["kenttarova", "hyytiala"]}

UPD = os.path.join(HERE, "data", "updraft")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")


def layer_mask(height_agl, cbh, top, depth):
    """(time, height) mask of gates between cbh - depth and cbh - top (cbh per time)."""
    h = np.asarray(height_agl)[None, :]
    c = np.asarray(cbh)[:, None]
    return (h >= c - depth) & (h <= c - top) & (h > 50)


def hourly_cbh(ds_in, hours):
    return pd.Series(ds_in["cloud_base_height"].values.astype(float), index=hours) \
        if "cloud_base_height" in ds_in else pd.Series(np.nan, index=hours)


# ----------------------------------------------------------------------------- ARM
def arm_site(site, hours, cbh_h, a):
    rows = []
    for stream in ARM_STREAMS.get(site, []):
        for f in sorted(glob.glob(os.path.join(UPD, "arm", stream, "*.nc")) +
                        glob.glob(os.path.join(UPD, "arm", stream, "*.cdf"))):
            try:
                with xr.open_dataset(f) as ds:
                    t = pd.DatetimeIndex(ds["time"].values)
                    hr = t.floor("h")
                    if not hr.isin(hours).any():
                        continue
                    z = ds["height"].values.astype(float)
                    cb = cbh_h.reindex(hr).values
                    if "dl_cbh" in ds:
                        cb = np.where(np.isfinite(cb), cb, ds["dl_cbh"].values.astype(float))
                    m = layer_mask(z, cb, a.top, a.depth)
                    thr = float(ds["snr_threshold"].values) if "snr_threshold" in ds else 0.0
                    if "snr" in ds:
                        m &= ds["snr"].values > thr
                    var = np.where(m, ds["w_variance"].values, np.nan)
                    wmed = np.where(m, ds["w"].values, np.nan)
                    var[var < 0] = np.nan
                    with np.errstate(all="ignore"):
                        v10, w10 = np.nanmean(var, axis=1), np.nanmean(wmed, axis=1)
                    cbq = (ds["cbw_75"].values - ds["cbw_25"].values) / 1.349 if "cbw_25" in ds else np.full(len(t), np.nan)
                    rows.append(pd.DataFrame(dict(hour=hr, var=v10, w=w10, cbs=cbq)))
            except Exception as ex:                                                  # noqa: BLE001
                print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
        if rows:
            break                                   # first stream with files (C1 before E37 handled at download)
    if not rows:
        return None
    df = pd.concat(rows)
    df = df[df.hour.isin(hours)]
    g = df.dropna(subset=["var", "w"]).groupby("hour")
    out = pd.DataFrame({"sigma_w": np.sqrt(g["var"].mean() + g["w"].var(ddof=0).fillna(0)),
                        "w_mean": g["w"].mean(), "n": g.size()})
    out["sigma_w_cb"] = df.groupby("hour")["cbs"].median()
    return out.reindex(hours), None, "arm_dlprofwstats_below_cb"


# ----------------------------------------------------------------------------- Cloudnet
def cloudnet_site(site, hours, cbh_h, a):
    cid = CLOUDNET_SITES.get(site)
    files = sorted(glob.glob(os.path.join(UPD, "cloudnet", cid or "-", "*.nc")))
    if not files:
        return None
    alt = SITE_COORDS[site][2]
    prof = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                t = pd.DatetimeIndex(ds["time"].values)
                hr = t.floor("h")
                keep = hr.isin(hours)
                if not keep.any():
                    continue
                v = ds["v"].values.astype(float)[keep]
                if "height" in ds:
                    z = ds["height"].values.astype(float) - float(ds["altitude"].values if "altitude" in ds else alt)
                else:
                    z = ds["range"].values.astype(float)
                cb = cbh_h.reindex(hr[keep]).values
                m = layer_mask(z, cb, a.top, a.depth)
                with np.errstate(all="ignore"):
                    wl = np.nanmean(np.where(m, v, np.nan), axis=1)
                prof.append(pd.DataFrame(dict(hour=hr[keep], w=wl)))
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
    if not prof:
        return None
    df = pd.concat(prof).dropna()
    g = df.groupby("hour")["w"]
    out = pd.DataFrame({"sigma_w": g.std(), "w_mean": g.mean(), "n": g.size()}).reindex(hours)
    out["sigma_w_cb"] = np.nan
    W = np.full((len(hours), a.max_samples), np.nan)
    pos = {h: i for i, h in enumerate(hours)}
    rng = np.random.default_rng(0)
    for h, s in g:
        x = s.values
        if len(x) > a.max_samples:
            x = rng.choice(x, a.max_samples, replace=False)
        W[pos[h], :len(x)] = x
    return out, W, "cloudnet_doppler_lidar_below_cb"


# ----------------------------------------------------------------------------- Cloudnet radar
def radar_site(site, hours, cbh_h, a):
    """Cloud-radar Doppler velocity just above cloud base (Cloudnet categorize, 30-s grid).

    Gates from cloud base to --radar-depth m above it that are classified as cloud droplets (category bit 0)
    and not falling hydrometeors (bit 1), with Z below --zmax dBZ and no rain at the surface: there the
    droplets fall at a few cm/s, so v ~ w. Per 30-s step the layer-mean v and the mean v_sigma (spread
    within the 30 s); per hour sigma_w = sqrt(var of the 30-s means + mean v_sigma^2)."""
    cid = CLOUDNET_RADAR_SITES.get(site)
    files = sorted(glob.glob(os.path.join(UPD, "cloudnet_categorize", cid or "-", "*.nc")))
    if not files:
        return None
    rows = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                t = pd.DatetimeIndex(ds["time"].values)
                hr = t.floor("h")
                keep = np.asarray(hr.isin(hours))
                if not keep.any():
                    continue
                alt = float(np.nanmedian(ds["altitude"].values))
                z = ds["height"].values.astype(float) - alt
                cb = cbh_h.reindex(hr[keep]).values
                h = z[None, :]
                m = (h >= cb[:, None]) & (h <= cb[:, None] + a.radar_depth)
                cat = ds["category_bits"].values[keep].astype(np.int64)
                m &= (cat & 1) > 0
                m &= (cat & 2) == 0
                m &= ds["Z"].values[keep] < a.zmax
                if "insect_prob" in ds:
                    m &= ~(ds["insect_prob"].values[keep] > 0.5)
                if "rain_detected" in ds:
                    m &= (ds["rain_detected"].values[keep] == 0)[:, None]
                v = np.where(m, ds["v"].values[keep], np.nan)
                vs = np.where(m, ds["v_sigma"].values[keep], np.nan) if "v_sigma" in ds else np.full(v.shape, np.nan)
                rows.append(pd.DataFrame(dict(hour=hr[keep], w=np.nanmean(v, axis=1),
                                              vs2=np.nanmean(vs ** 2, axis=1))))
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
    if not rows:
        return None
    df = pd.concat(rows).dropna(subset=["w"])
    g = df.groupby("hour")
    out = pd.DataFrame({"sigma_w": np.sqrt(g["w"].var(ddof=0) + g["vs2"].mean().fillna(0)),
                        "w_mean": g["w"].mean(), "n": g.size()}).reindex(hours)
    out["sigma_w_cb"] = g["w"].std().reindex(hours)          # without the sub-30-s part, as a lower bound
    return out, None, "cloudnet_radar_above_cb"


def fallback_site(site, hours):
    """Constant sigma_w for every hour: median of the pooled hourly sigma_w of the FALLBACK lidar sites."""
    pool = []
    for s in FALLBACK.get(site, []):
        p = os.path.join(HERE, "updrafts", f"{s}_w_cloudbase.nc")
        if os.path.exists(p):
            with xr.open_dataset(p) as d:
                pool.append(d["sigma_w"].values)
    x = np.concatenate(pool) if pool else np.array([])
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return None
    q10, q50, q90 = np.percentile(x, [10, 50, 90])
    out = pd.DataFrame({"sigma_w": q50, "w_mean": 0.0, "n": len(x), "sigma_w_cb": np.nan}, index=hours)
    print(f"    pooled {len(x)} hours from {','.join(FALLBACK[site])}: sigma_w 10/50/90 % = "
          f"{q10:.2f}/{q50:.2f}/{q90:.2f} (use --sigma-w {q10:.2f} / {q90:.2f} in weight_updrafts.py as bounds)")
    return out, None, f"regime_fallback_{'+'.join(FALLBACK[site])}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--top", type=float, default=50.0, help="layer top, m below cloud base")
    ap.add_argument("--depth", type=float, default=300.0, help="layer bottom, m below cloud base")
    ap.add_argument("--min-n", type=int, default=3, help="minimum 10-min intervals (ARM) or 30 profiles (Cloudnet)")
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--radar-depth", type=float, default=150.0, help="radar layer, m above cloud base")
    ap.add_argument("--zmax", type=float, default=-20.0, help="max radar reflectivity (dBZ) for non-drizzling gates")
    a = ap.parse_args()
    sites = a.sites or (sorted(set(ARM_STREAMS) | set(CLOUDNET_SITES) | set(CLOUDNET_RADAR_SITES) - set(FALLBACK))
                        + sorted(FALLBACK))                 # fallbacks last: they pool the lidar sites' results
    os.makedirs(os.path.join(HERE, "updrafts"), exist_ok=True)
    for site in sites:
        p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
        if not os.path.exists(p):
            continue
        with xr.open_dataset(p) as ds:
            hours = pd.DatetimeIndex(ds["hour"].values)
            cbh = hourly_cbh(ds, hours)
        if site in ARM_STREAMS:
            r = arm_site(site, hours, cbh, a)
        elif site in CLOUDNET_SITES:
            r = cloudnet_site(site, hours, cbh, a)
        elif site in FALLBACK:
            r = fallback_site(site, hours)
        else:
            r = radar_site(site, hours, cbh, a)
        if r is None:
            print(f"{site}: no lidar files")
            continue
        out, W, src = r
        nmin = 30 if W is not None else (20 if src.startswith("cloudnet_radar") else a.min_n)  # 20 x 30 s = 10 min
        ok = out["n"].fillna(0) >= nmin
        out.loc[~ok, ["sigma_w", "w_mean"]] = np.nan
        dv = {"sigma_w": ("hour", out.sigma_w.values, {"units": "m s-1", "long_name": f"w std {a.top:.0f}-{a.depth:.0f} m below cloud base"}),
              "w_mean": ("hour", out.w_mean.values, {"units": "m s-1"}),
              "n_samples": ("hour", out.n.fillna(0).astype(int).values),
              "sigma_w_cb": ("hour", out.sigma_w_cb.values, {"units": "m s-1", "long_name": "cloud-base w IQR / 1.349 (ARM VAP)"})}
        if W is not None:
            dv["w"] = (("hour", "sample"), W, {"units": "m s-1"})
        xr.Dataset(dv, coords={"hour": hours.values},
                   attrs={"w_source": src, "layer": f"{a.top}-{a.depth} m below cloud base"}
                   ).to_netcdf(os.path.join(HERE, "updrafts", f"{site}_w_radar.nc" if site == "zeppelin"
                                            else f"{site}_w_cloudbase.nc"))
        s = out.sigma_w[ok]
        extra = f", sigma_w_cb median {np.nanmedian(out.sigma_w_cb):.2f}" if out.sigma_w_cb.notna().any() else ""
        print(f"{site:16s} {src:32s} {ok.sum():5d}/{len(hours)} hours, sigma_w median {s.median():.2f} "
              f"(10-90 %: {s.quantile(.1):.2f}-{s.quantile(.9):.2f}), w_mean median {out.w_mean[ok].median():+.2f}{extra}")


if __name__ == "__main__":
    main()
