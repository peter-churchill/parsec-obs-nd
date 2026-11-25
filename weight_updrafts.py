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

The same weights give the hour's Smax_parcel and Dact_parcel_nm (dry activation diameter,
2 x the run's minimum activation radius). Ntot_input is the total number of the parcel's input
size distribution and sigma_w the hour's updraft spread (from the w file, or the std of its samples).

Examples
--------
python weight_updrafts.py --site ena --sigma-w 0.3
python weight_updrafts.py --site ena --w-file updrafts/ena_w_cloudbase.nc --weighting flux
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
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
    if "r_act_min_nm" in df:
        ract = df.pivot(index="hour_index", columns="w", values="r_act_min_nm").reindex(nd.index)
        ract = ract.where(ract > 0)
    else:
        ract = smax * np.nan
    ntot = df.groupby("hour_index")["N_input_cm3"].first().reindex(nd.index)
    hours = df.groupby("hour_index")["hour"].first().reindex(nd.index)
    print(f"{site}: {len(files)} chunks, {nd.shape[0]} hours x {nd.shape[1]} updrafts, {int(bad.sum())} runs not ok")
    return nd, smax, ract, ntot, hours


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
    ap.add_argument("--sigma-only", action="store_true", help="ignore w samples in --w-file, use sigma_w / w_mean")
    ap.add_argument("--out", help="default results/<site>_parcel_nd_<tag>.nc")
    a = ap.parse_args()

    nd, smax, ract, ntot, hours = load_runs(a.rundir, a.site, a.criterion)
    wgrid = nd.columns.to_numpy(float)
    inp = xr.open_dataset(a.inputs or os.path.join(HERE, "inputs", f"{a.site}_parcel_inputs.nc"))
    nh = len(nd)
    out_nd, out_smax, out_ract, out_sig = (np.full(nh, np.nan) for _ in range(4))
    n_used = np.zeros(nh, int)
    clamped = np.full(nh, np.nan)
    wsrc = np.array(["none"] * nh, dtype=object)

    def apply(i, w, p):
        """Weighted Nd, Smax and activation radius of hour i over updrafts w with PDF weights p."""
        f = p * (w if a.weighting == "flux" else 1.0)
        for out, tab in ((out_nd, nd), (out_smax, smax), (out_ract, ract)):
            v = nd_of_w(w, wgrid, tab.iloc[i].to_numpy(float))
            ok = np.isfinite(v)
            out[i] = np.sum(v[ok] * f[ok]) / np.sum(f[ok]) if ok.any() else np.nan

    if a.sigma_w is not None:
        wq, pq = gaussian_positive(a.sigma_w)
        for i in range(nh):
            apply(i, wq, pq)
            clamped[i] = pq[(wq < wgrid[0]) | (wq > wgrid[-1])].sum()
        out_sig[:] = a.sigma_w
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
            hs = str(wf["hour_source"].values[k]) if "hour_source" in wf else str(wf.attrs.get("w_source", ""))
            wall = wf["w"].values[k] if ("w" in wf and not a.sigma_only) else np.array([])
            wall = wall[np.isfinite(wall)]
            if "sigma_w" in wf and np.isfinite(wf["sigma_w"].values[k]):
                out_sig[i] = float(wf["sigma_w"].values[k])
            elif len(wall) > 2:
                out_sig[i] = float(np.std(wall))
            w = wall[wall > 0]
            if len(w) >= a.min_samples:                 # samples where the hour has enough, else the Gaussian
                apply(i, w, np.ones_like(w))
                n_used[i] = len(w)
                clamped[i] = np.mean((w < wgrid[0]) | (w > wgrid[-1]))
                wsrc[i] = f"{hs}_samples"
            elif "sigma_w" in wf:
                sg = float(wf["sigma_w"].values[k])
                m = float(wf["w_mean"].values[k]) if "w_mean" in wf else 0.0
                if not np.isfinite(sg) or sg <= 0:
                    continue
                wq, pq = gaussian_positive(sg, m)
                if wq is None:
                    continue
                apply(i, wq, pq)
                clamped[i] = pq[(wq < wgrid[0]) | (wq > wgrid[-1])].sum()
                wsrc[i] = f"{hs}_sigma"
        tag = f"{os.path.splitext(os.path.basename(a.w_file))[0]}_{a.weighting}"

    hix = nd.index.to_numpy()
    ds = xr.Dataset(
        {"Nd_parcel": ("hour", out_nd, {"units": "cm-3", "long_name": f"updraft-weighted parcel Nd ({a.weighting}), criterion {a.criterion}"}),
         "Nd_w": (("hour", "w"), nd.to_numpy(), {"units": "cm-3", "long_name": "parcel Nd on the updraft grid"}),
         "Smax_w": (("hour", "w"), smax.to_numpy(), {"units": "%"}),
         "Smax_parcel": ("hour", out_smax, {"units": "%", "long_name": f"updraft-weighted ({a.weighting}) parcel Smax"}),
         "Dact_parcel_nm": ("hour", 2 * out_ract, {"units": "nm", "long_name": f"updraft-weighted ({a.weighting}) dry activation diameter"}),
         "Ntot_input": ("hour", ntot.to_numpy(float), {"units": "cm-3", "long_name": "total number of the parcel input size distribution"}),
         "sigma_w": ("hour", out_sig, {"units": "m s-1"}),
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
