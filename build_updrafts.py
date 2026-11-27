#!/usr/bin/env python
"""Hourly updraft statistics just below cloud base from Doppler lidar, for weight_updrafts.py.

For each hour of inputs/<site>_parcel_inputs.nc, vertical velocity is taken in a layer below cloud base,
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
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
  Radar sigma_w is divided by RADAR_TO_LIDAR (1.19, the median radar/lidar ratio where both exist), since the
  in-cloud radar layer is more turbulent than the sub-cloud lidar layer; sigma_w_radar_raw keeps the original.
Puijo, Pallas: station climatology from paper_updrafts.py (the paper's Zenodo / Kenttarova lidar data), kept
  if present; otherwise (or with --fallback) one sigma_w for all hours, the median of the boreal lidar sites' hours.

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
from sites import SITE_COORDS, SITES  # noqa: E402
from updraft_download import ARM_STREAMS, CLOUDNET_RADAR_SITES, CLOUDNET_SITES  # noqa: E402

# sites with neither lidar nor radar on their cloud days: per-regime sigma_w pooled from these lidar sites
FALLBACK = {"puijo": ["hyytiala", "kenttarova"], "pallas": ["kenttarova", "hyytiala"]}
# Radar sigma_w (in cloud, just above base) / lidar sigma_w (just below base) over the 1030 hours with both at
# Hyytiala, Granada, Kenttarova, Melpitz, Bucharest and Cabauw (2026-11-20 run): hour-weighted median ratio 1.19.
# Radar values are divided by it (--radar-scale) so all sites refer to the sub-cloud lidar layer.
RADAR_TO_LIDAR = 1.19

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
        x = s.values - s.values.mean()          # mean below cloud base is ~0; Cloudnet means are biased (see main)
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
                cat = ds["category_bits"].values[keep].astype(np.int64)
                if not np.isfinite(cb).any():           # in-cloud stations: lowest cloud-droplet gate per profile
                    liq = (cat & 1) > 0
                    first = np.where(liq.any(axis=1), np.argmax(liq, axis=1), -1)
                    cb = np.where(first >= 0, z[np.clip(first, 0, None)], np.nan)
                h = z[None, :]
                m = (h >= cb[:, None]) & (h <= cb[:, None] + a.radar_depth)
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
    out["sigma_w_radar_raw"] = out["sigma_w"]
    out["sigma_w"] = out["sigma_w"] / a.radar_scale
    return out, None, "cloudnet_radar_above_cb"


def aircraft_site(site, hours):
    """Aircraft: the 1-Hz vertical wind measured during each cloud pass (aircraft_build.py --met-dir), as is
    (not demeaned: the mean updraft of a near-base pass is part of the signal). Pass sigma_w, w_mean, n."""
    root = os.environ.get("OBS_REPO")
    if not root:
        print(f"{site}: set OBS_REPO to read the aircraft built file")
        return None
    p = os.path.join(root, SITES[site]["built"])
    if not os.path.exists(p):
        print(f"{site}: built file missing ({p})")
        return None
    with xr.open_dataset(p) as ds:
        if "w" not in ds:
            print(f"{site}: built file has no w (rebuild with aircraft_build.py --met-dir)")
            return None
        t = pd.DatetimeIndex(ds["time"].values)
        Wb = ds["w"].values.astype(float)
    k = pd.Series(np.arange(len(t)), index=t)
    k = k[~k.index.duplicated()].reindex(hours)
    W = np.full((len(hours), Wb.shape[1]), np.nan)
    have = k.notna().to_numpy()
    W[have] = Wb[k[have].astype(int).to_numpy()]
    n = np.isfinite(W).sum(axis=1)
    out = pd.DataFrame({"sigma_w": np.nanstd(np.where(n[:, None] > 0, W, 0.0), axis=1),
                        "w_mean": np.nanmean(np.where(n[:, None] > 0, W, 0.0), axis=1), "n": n,
                        "sigma_w_cb": np.nan}, index=hours)
    out.loc[n == 0, ["sigma_w", "w_mean"]] = np.nan
    out["nmin"] = 10
    return out, W, "aircraft_pass_w"


def fallback_site(site, hours):
    """Constant sigma_w for every hour: median of the pooled hourly sigma_w of the FALLBACK lidar sites."""
    pool = []
    for s in FALLBACK.get(site, []):
        p = os.path.join(HERE, "updrafts", f"{s}_w_cloudbase.nc")
        if os.path.exists(p):
            with xr.open_dataset(p) as d:                  # lidar hours only (radar-filled hours excluded)
                pool.append(d["sigma_w_lidar"].values if "sigma_w_lidar" in d else d["sigma_w"].values)
    x = np.concatenate(pool) if pool else np.array([])
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return None
    q10, q50, q90 = np.percentile(x, [10, 50, 90])
    out = pd.DataFrame({"sigma_w": q50, "w_mean": 0.0, "n": len(x), "sigma_w_cb": np.nan}, index=hours)
    print(f"    pooled {len(x)} hours from {','.join(FALLBACK[site])}: sigma_w 10/50/90 % = "
          f"{q10:.2f}/{q50:.2f}/{q90:.2f} (use --sigma-w {q10:.2f} / {q90:.2f} in weight_updrafts.py as bounds)")
    return out, None, f"regime_fallback_{'+'.join(FALLBACK[site])}"


def combine(lid, rad, hours):
    """Lidar (below cloud base) where it has >= 30 profiles in the hour, else radar (above cloud base) with
    >= 20 30-s steps. Both sigma_w are kept for the hours that have both."""
    if lid is None and rad is None:
        return None
    if rad is None:
        return lid
    if lid is None:
        return rad
    L, W, _ = lid
    R = rad[0]
    okL = L["n"].fillna(0) >= 30
    okR = R["n"].fillna(0) >= 20
    out = L.copy()
    useR = ~okL & okR
    for c in ("sigma_w", "w_mean", "n", "sigma_w_cb"):
        out.loc[useR, c] = R.loc[useR, c]
    out["nmin"] = np.where(useR, 20, 30)
    out["hour_source"] = np.where(okL, "lidar", np.where(okR, "radar", "none"))
    out["sigma_w_lidar"] = L.sigma_w.where(okL)
    out["sigma_w_radar"] = R["sigma_w_radar_raw"].where(okR)
    if W is not None:
        W[useR.values] = np.nan
    return out, W, "lidar+radar"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--top", type=float, default=50.0, help="layer top, m below cloud base")
    ap.add_argument("--depth", type=float, default=300.0, help="layer bottom, m below cloud base")
    ap.add_argument("--min-n", type=int, default=3, help="minimum 10-min intervals (ARM) or 30 profiles (Cloudnet)")
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--radar-depth", type=float, default=150.0, help="radar layer, m above cloud base")
    ap.add_argument("--zmax", type=float, default=-20.0, help="max radar reflectivity (dBZ) for non-drizzling gates")
    ap.add_argument("--radar-scale", type=float, default=RADAR_TO_LIDAR,
                    help="divide radar sigma_w by this (radar/lidar ratio where both exist; 1 = no scaling)")
    ap.add_argument("--fallback", action="store_true",
                    help="Puijo/Pallas: overwrite the paper_updrafts.py files with the pooled lidar fallback")
    a = ap.parse_args()
    aircraft = sorted(k for k, v in SITES.items() if v["kind"] == "aircraft")
    sites = a.sites or (sorted(set(ARM_STREAMS) | set(CLOUDNET_SITES) | set(CLOUDNET_RADAR_SITES) | set(aircraft)
                               - set(FALLBACK))
                        + sorted(FALLBACK))                 # fallbacks last: they pool the lidar sites' results
    os.makedirs(os.path.join(HERE, "updrafts"), exist_ok=True)
    for site in sites:
        p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
        if not os.path.exists(p):
            continue
        with xr.open_dataset(p) as ds:
            hours = pd.DatetimeIndex(ds["hour"].values)
            cbh = hourly_cbh(ds, hours)
        if SITES.get(site, {}).get("kind") == "aircraft":
            r = aircraft_site(site, hours)
        elif site in ARM_STREAMS:
            r = arm_site(site, hours, cbh, a)
        elif site in FALLBACK:
            q = os.path.join(HERE, "updrafts", f"{site}_w_cloudbase.nc")
            if os.path.exists(q) and not a.fallback:
                with xr.open_dataset(q) as d:
                    paper = not str(d.attrs.get("w_source", "")).startswith("regime_fallback")
                if paper:                                   # from paper_updrafts.py: keep it
                    print(f"{site}: keeping {os.path.basename(q)} from paper_updrafts.py (--fallback to replace)")
                    continue
            r = fallback_site(site, hours)
        elif site in CLOUDNET_SITES:
            r = combine(cloudnet_site(site, hours, cbh, a), radar_site(site, hours, cbh, a), hours)
        else:
            r = radar_site(site, hours, cbh, a)
        if r is None:
            print(f"{site}: no lidar or radar files")
            continue
        out, W, src = r
        if "nmin" not in out:
            out["nmin"] = 30 if W is not None else (20 if src.startswith("cloudnet_radar") else a.min_n)
        ok = out["n"].fillna(0) >= out["nmin"]
        out.loc[~ok, ["sigma_w", "w_mean"]] = np.nan
        if W is not None:
            W[~ok.values] = np.nan
        # Cloudnet lidar/radar hourly means come out at -0.1 to -0.2 m/s (-0.6 at Schneefernerhaus): drizzle/ice
        # fall speeds or terrain, not the mean updraft below a stratiform cloud base, which is ~0 (as the ARM
        # lidars show). The Gaussian therefore gets mean 0; the observed mean is kept as w_mean_obs.
        wm = (out.w_mean * 0.0 if src.startswith(("cloudnet", "lidar+radar")) else out.w_mean).where(ok)
        hsrc = out["hour_source"].values.astype(object) if "hour_source" in out else np.where(ok, src, "none").astype(object)
        dv = {"sigma_w": ("hour", out.sigma_w.values, {"units": "m s-1", "long_name": "w std near cloud base"}),
              "w_mean": ("hour", wm.values, {"units": "m s-1", "long_name": "mean w used for the PDF"}),
              "w_mean_obs": ("hour", out.w_mean.values, {"units": "m s-1", "long_name": "observed hourly mean w"}),
              "n_samples": ("hour", out.n.fillna(0).astype(int).values),
              "sigma_w_cb": ("hour", out.sigma_w_cb.values, {"units": "m s-1",
                             "long_name": "check: ARM cloud-base IQR/1.349, or radar without sub-30-s part"}),
              "hour_source": ("hour", hsrc)}
        for k in ("sigma_w_lidar", "sigma_w_radar", "sigma_w_radar_raw"):
            if k in out:
                dv[k] = ("hour", out[k].values, {"units": "m s-1"})
        if W is not None:
            dv["w"] = (("hour", "sample"), W, {"units": "m s-1", "long_name": "w minus hourly mean"})
        xr.Dataset(dv, coords={"hour": hours.values},
                   attrs={"w_source": src, "lidar_layer": f"{a.top}-{a.depth} m below cloud base",
                          "radar_layer": f"0-{a.radar_depth} m above cloud base, Z < {a.zmax} dBZ"}
                   ).to_netcdf(os.path.join(HERE, "updrafts", f"{site}_w_radar.nc" if site == "zeppelin"
                                            else f"{site}_w_cloudbase.nc"))
        sw = out.sigma_w[ok]
        extra = f", check {np.nanmedian(out.sigma_w_cb):.2f}" if out.sigma_w_cb.notna().any() else ""
        print(f"{site:16s} {src:32s} {ok.sum():5d}/{len(hours)} hours, sigma_w median {sw.median():.2f} "
              f"(10-90 %: {sw.quantile(.1):.2f}-{sw.quantile(.9):.2f}), w_mean_obs median "
              f"{out.w_mean[ok].median():+.2f}{extra}", flush=True)
        if "hour_source" in out:
            print("    hours by source: " + ", ".join(f"{k} {v}" for k, v in pd.Series(hsrc[ok.values]).value_counts().items()))
        if "sigma_w_lidar" in out:
            both = out.sigma_w_lidar.notna() & out.sigma_w_radar.notna()
            if both.sum() >= 10:
                print(f"    lidar vs radar sigma_w (radar unscaled) in {both.sum()} hours with both: median ratio radar/lidar "
                      f"{np.median(out.sigma_w_radar[both] / out.sigma_w_lidar[both]):.2f}, r = "
                      f"{np.corrcoef(out.sigma_w_radar[both], out.sigma_w_lidar[both])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
