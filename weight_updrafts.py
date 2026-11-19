#!/usr/bin/env python
"""Combine the PARSEC runs of a site and weight Nd(w) by each hour's updraft distribution.

Reads runs/<site>/chunk_*.csv (from parsec_run.py), builds Nd(hour, w)
for the chosen activation criterion, and computes one parcel Nd per hour:

  samples  --w-file has a variable `w` (hour, sample) in m s-1, e.g. Doppler-lidar
           vertical velocity just below cloud base. Nd_hour = mean of Nd(w) over the
           positive samples ("pdf" weighting) or the w-weighted mean ("flux" weighting).
           Nd(w) is interpolated linearly in log w - log Nd between the grid points;
           below the grid Nd(w) is extrapolated in log-log, above it held constant (frac_w_outside).
  sigma    --w-file has `sigma_w` (hour) (and optionally `w_mean`); the hour's w PDF
           is a Gaussian, and Nd is integrated over its positive half the same way.
  fixed    --sigma-w S applies one sigma_w to every hour (sensitivity runs while the
           lidar data are not in yet).

Hours without updraft information get NaN, with w_source = "none". The output keeps
the full Nd(hour, w) table so other weightings can be tried without rerunning PARSEC.

Examples
--------
python weight_updrafts.py --site ena --sigma-w 0.3
python weight_updrafts.py --site ena --w-file updrafts/ena_w_cloudbase.nc --weighting flux
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runs(rundir, site, criterion):
    files = sorted(glob.glob(os.path.join(rundir, site, "chunk_*.csv")))
    if not files:
        raise SystemExit(f"no chunk files in {os.path.join(rundir, site)}")
    df = pd.concat([pd.read_csv(f, parse_dates=["hour"]) for f in files], ignore_index=True)
    df = df.drop_duplicates(["hour_index", "w"], keep="last")
    bad = df["status"] != "ok"
    col = f"Nd_c{criterion}"
    if col not in df:
        df[col] = df[f"AF_c{criterion}"] * df["N_input_cm3"]
    df.loc[bad, col] = np.nan
    nd = df.pivot(index="hour_index", columns="w", values=col).sort_index()
    smax = df.pivot(index="hour_index", columns="w", values="Smax_pct").reindex(nd.index)
    hours = df.groupby("hour_index")["hour"].first().reindex(nd.index)
    print(f"{site}: {len(files)} chunks, {nd.shape[0]} hours x {nd.shape[1]} updrafts, {int(bad.sum())} runs not ok")
    return nd, smax, hours


def nd_of_w(w, wgrid, ndrow):
    """Nd at updrafts w by log-log interpolation of one hour's Nd(w) curve.

    Below the lowest grid updraft the curve is extrapolated with the log-log slope of
    the two lowest points (Nd keeps falling as w -> 0 instead of staying at Nd(w_min));
    above the highest it is held constant.
    """
    ok = np.isfinite(ndrow) & (ndrow > 0)
    if ok.sum() < 2:
        return np.full(np.shape(w), np.nan)
    lx, ly = np.log(wgrid[ok]), np.log(ndrow[ok])
    lw = np.log(np.clip(np.asarray(w, float), 1e-4, wgrid[ok][-1]))
    out = np.interp(lw, lx, ly)
    lo = lw < lx[0]
    if lo.any():
        slope = (ly[1] - ly[0]) / (lx[1] - lx[0])
        out[lo] = ly[0] + slope * (lw[lo] - lx[0])
    return np.exp(out)


def weighted(nd_w, w, weighting):
    if weighting == "flux":
        return np.sum(nd_w * w) / np.sum(w)
    return np.mean(nd_w)


def gaussian_positive(sigma, mean=0.0, n=400):
    """Quadrature points/weights over the positive half of N(mean, sigma)."""
    hi = mean + 5 * sigma
    if hi <= 0:
        return None, None
    w = np.linspace(max(1e-4, 0.0), hi, n)
    p = np.exp(-0.5 * ((w - mean) / sigma) ** 2)
    return w, p / p.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--rundir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--inputs", help="default inputs/<site>_parcel_inputs.nc")
    ap.add_argument("--criterion", type=int, default=3, choices=[1, 2, 3, 4],
                    help="PARSEC activation criterion (3 = wet radius > critical radius)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--w-file", help="NetCDF with w(hour, sample) or sigma_w(hour)")
    src.add_argument("--sigma-w", type=float, help="one sigma_w (m s-1) for all hours")
    ap.add_argument("--weighting", choices=["pdf", "flux"], default="pdf")
    ap.add_argument("--min-samples", type=int, default=30, help="minimum positive w samples per hour")
    ap.add_argument("--out", help="default results/<site>_parcel_nd_<tag>.nc")
    a = ap.parse_args()

    nd, smax, hours = load_runs(a.rundir, a.site, a.criterion)
    wgrid = nd.columns.to_numpy(float)
    inp = xr.open_dataset(a.inputs or os.path.join(HERE, "inputs", f"{a.site}_parcel_inputs.nc"))
    nh = len(nd)
    out_nd = np.full(nh, np.nan)
    n_used = np.zeros(nh, int)
    clamped = np.full(nh, np.nan)
    wsrc = np.array(["none"] * nh, dtype=object)

    if a.sigma_w is not None:
        wq, pq = gaussian_positive(a.sigma_w)
        for i in range(nh):
            v = nd_of_w(wq, wgrid, nd.iloc[i].to_numpy())
            out_nd[i] = np.sum(v * pq * (wq if a.weighting == "flux" else 1)) / np.sum(pq * (wq if a.weighting == "flux" else 1))
            clamped[i] = pq[(wq < wgrid[0]) | (wq > wgrid[-1])].sum()
        wsrc[:] = f"fixed_sigma_{a.sigma_w:g}"
        tag = f"sigma{a.sigma_w:g}_{a.weighting}"
    else:
        wf = xr.open_dataset(a.w_file)
        wf_hours = pd.DatetimeIndex(wf["hour"].values)
        pos = {h: k for k, h in enumerate(wf_hours)}
        for i in range(nh):
            k = pos.get(pd.Timestamp(hours.iloc[i]))
            if k is None:
                continue
            row = nd.iloc[i].to_numpy()
            if "w" in wf:
                w = wf["w"].values[k]
                w = w[np.isfinite(w) & (w > 0)]
                if len(w) < a.min_samples:
                    continue
                out_nd[i] = weighted(nd_of_w(w, wgrid, row), w, a.weighting)
                n_used[i] = len(w)
                clamped[i] = np.mean((w < wgrid[0]) | (w > wgrid[-1]))
                wsrc[i] = str(wf.attrs.get("w_source", "samples"))
            elif "sigma_w" in wf:
                s = float(wf["sigma_w"].values[k])
                m = float(wf["w_mean"].values[k]) if "w_mean" in wf else 0.0
                if not np.isfinite(s) or s <= 0:
                    continue
                wq, pq = gaussian_positive(s, m)
                if wq is None:
                    continue
                v = nd_of_w(wq, wgrid, row)
                f = wq if a.weighting == "flux" else 1
                out_nd[i] = np.sum(v * pq * f) / np.sum(pq * f)
                clamped[i] = pq[(wq < wgrid[0]) | (wq > wgrid[-1])].sum()
                wsrc[i] = str(wf.attrs.get("w_source", "sigma_w"))
        tag = f"{os.path.splitext(os.path.basename(a.w_file))[0]}_{a.weighting}"

    hix = nd.index.to_numpy()
    ds = xr.Dataset(
        {"Nd_parcel": ("hour", out_nd, {"units": "cm-3", "long_name": f"updraft-weighted parcel Nd ({a.weighting}), criterion {a.criterion}"}),
         "Nd_w": (("hour", "w"), nd.to_numpy(), {"units": "cm-3", "long_name": "parcel Nd on the updraft grid"}),
         "Smax_w": (("hour", "w"), smax.to_numpy(), {"units": "%"}),
         "n_w_samples": ("hour", n_used),
         "frac_w_outside": ("hour", clamped, {"long_name": "fraction of the w PDF outside the run grid (extrapolated below, held above)"}),
         "w_source": ("hour", wsrc),
         "Nd_obs": ("hour", inp["Nd_obs"].values[hix], {"units": "cm-3"}),
         "N100": ("hour", inp["N100"].values[hix], {"units": "cm-3"}),
         "comp_source": ("hour", inp["comp_source"].values[hix]),
         "thermo_source": ("hour", inp["thermo_source"].values[hix])},
        coords={"hour": pd.DatetimeIndex(hours.to_numpy()).values, "w": ("w", wgrid, {"units": "m s-1"})},
        attrs={"site": a.site, "criterion": a.criterion, "weighting": a.weighting, "w_input": a.w_file or f"sigma_w={a.sigma_w}"})
    out = a.out or os.path.join(HERE, "results", f"{a.site}_parcel_nd_{tag}.nc")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    ds.to_netcdf(out)
    ok = np.isfinite(out_nd) & np.isfinite(ds["Nd_obs"].values) & (ds["Nd_obs"].values > 0)
    if ok.any():
        r = np.log10(out_nd[ok] / ds["Nd_obs"].values[ok])
        print(f"{a.site}: {ok.sum()} hours with parcel and observed Nd; median log10(parcel/obs) = {np.median(r):+.2f} "
              f"(IQR {np.percentile(r, 25):+.2f} to {np.percentile(r, 75):+.2f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
