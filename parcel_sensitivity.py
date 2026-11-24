#!/usr/bin/env python
"""Free sensitivity tests on the existing parcel runs: activation criterion and updraft strength.

No new PARSEC runs: weight_updrafts.py is rerun on runs/<site>/ with
  criterion   1-4, PARSEC's activation criteria (mo_parsec_commons.f90):
                N1 wet radius > 1 um (droplet-sized: closest to what total-minus-interstitial inlets count)
                N2 critical supersaturation < peak supersaturation (equilibrium upper bound)
                N3 wet radius > critical radius (used so far)
                N4 every particle larger than the smallest activated one
              at the observed sigma_w;
  sigma_w x f the observed updraft file with sigma_w and the w samples multiplied by each --factors value
              (criterion --criterion), to see whether one scaling fixes the level.
Per site and variant, on hours with observed Nd, a parcel Nd and real T/P: n, median parcel/observed (IQR),
r(log Nd) and log-log slope; also the Spearman correlation of log(parcel/observed) with the hour's sigma_w
(does the error depend on turbulence?). --hours-file restricts to a screened hour set (site, hour), e.g. the
shallow-coupled or LWP-screened hours.

Writes results/parcel_sensitivity[/<tag>]/summary.csv and sensitivity.png.

python parcel_sensitivity.py --sites puijo pallas lasic ena
python parcel_sensitivity.py --hours-file results/cloudnet_quality/ideal_hours.csv --tag ideal
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))


def stats(o, p):
    ok = np.isfinite(o) & np.isfinite(p) & (o > 0) & (p > 0)
    o, p = o[ok], p[ok]
    if len(o) < 5:
        return dict(n=len(o), ratio=np.nan, q25=np.nan, q75=np.nan, r=np.nan, slope=np.nan)
    lo, lp = np.log10(o), np.log10(p)
    d = lp - lo
    return dict(n=len(o), ratio=10 ** np.median(d), q25=10 ** np.percentile(d, 25), q75=10 ** np.percentile(d, 75),
                r=np.corrcoef(lo, lp)[0, 1], slope=np.polyfit(lo, lp, 1)[0])


def weight(site, crit, wfile, wt, out, rundir):
    cmd = [sys.executable, os.path.join(HERE, "weight_updrafts.py"), "--site", site, "--rundir", rundir,
           "--criterion", str(crit), "--weighting", wt, "--w-file", wfile, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   {site} c{crit} {os.path.basename(wfile)}: failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:120]}")
        return None
    with xr.open_dataset(out) as d:
        return pd.DataFrame({"Nd_parcel": d["Nd_parcel"].values, "Nd_obs": d["Nd_obs"].values,
                             "thermo": d["thermo_source"].values.astype(str)},
                            index=pd.DatetimeIndex(d["hour"].values))


def scaled_wfile(wf, f, dest):
    with xr.open_dataset(wf) as d:
        d = d.load()
    d["sigma_w"] = d["sigma_w"] * f
    if "w" in d:
        d["w"] = d["w"] * f                       # samples are deviations about the hourly mean
    d.attrs["sigma_scale"] = f
    d.to_netcdf(dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--rundir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--weighting", choices=["pdf", "flux"], default="flux")
    ap.add_argument("--criterion", type=int, default=3, help="criterion for the sigma_w scaling")
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--hours-file", help="CSV with columns site, hour: only these hours")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", "parcel_sensitivity", a.tag)
    tmp = os.path.join(out, "weighted")
    os.makedirs(tmp, exist_ok=True)
    keep = None
    if a.hours_file:
        h = pd.read_csv(a.hours_file, parse_dates=["hour"])
        keep = {k: pd.DatetimeIndex(g["hour"]) for k, g in h.groupby("site")}
    sites = a.sites or sorted(d for d in os.listdir(a.rundir) if glob.glob(os.path.join(a.rundir, d, "chunk_*.csv")))
    if keep is not None:
        sites = [s for s in sites if s in keep]
    rows = []
    for site in sites:
        wf = os.path.join(HERE, "updrafts", f"{site}_w_cloudbase.nc")
        if not os.path.exists(wf):
            print(f"{site}: no updraft file; skipped")
            continue
        with xr.open_dataset(wf) as d:
            sig = pd.Series(d["sigma_w"].values, index=pd.DatetimeIndex(d["hour"].values))
        print(f"{site}", flush=True)
        variants = [(f"N{c}", c, 1.0) for c in (1, 2, 3, 4)] + \
                   [(f"N{a.criterion} sigma x{f:g}", a.criterion, f) for f in a.factors if f != 1.0]
        for name, c, f in variants:
            w = wf if f == 1.0 else scaled_wfile(wf, f, os.path.join(tmp, f"{site}_w_x{f:g}.nc"))
            d = weight(site, c, w, a.weighting, os.path.join(tmp, f"{site}_c{c}_x{f:g}.nc"), a.rundir)
            if d is None:
                continue
            d = d[d["thermo"] != "placeholder"]
            if keep is not None:
                d = d[d.index.isin(keep[site])]
            s = stats(d["Nd_obs"].to_numpy(float), d["Nd_parcel"].to_numpy(float))
            row = dict(site=site, variant=name, criterion=c, sigma_factor=f, **s)
            if name == f"N{a.criterion}":                       # does the error depend on turbulence?
                from scipy.stats import spearmanr
                lr = np.log10(d["Nd_parcel"] / d["Nd_obs"])
                sw = sig.reindex(d.index)
                ok = np.isfinite(lr) & np.isfinite(sw) & (sw.nunique() > 1)
                if ok.sum() > 10:
                    row["rho_ratio_sigma_w"] = spearmanr(sw[ok], lr[ok])[0]
            rows.append(row)
            print(f"   {name:16s} n {s['n']:5d}  ratio {s['ratio']:.2f} ({s['q25']:.2f}-{s['q75']:.2f})  "
                  f"r {s['r']:.2f}  slope {s['slope']:.2f}"
                  + (f"  rho(ratio, sigma_w) {row['rho_ratio_sigma_w']:+.2f}" if "rho_ratio_sigma_w" in row else ""),
                  flush=True)
    if not rows:
        raise SystemExit("nothing to compare")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.3g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    cmap = plt.get_cmap("tab20")
    for k, (site, g) in enumerate(t.groupby("site")):
        c = cmap(k % 20)
        cr = g[g["sigma_factor"] == 1.0].sort_values("criterion")
        axs[0].plot(cr["criterion"], cr["ratio"], "o-", color=c, label=site)
        axs[1].plot(cr["criterion"], cr["r"], "o-", color=c)
        sc = g[g["criterion"] == a.criterion].sort_values("sigma_factor")
        axs[2].plot(sc["sigma_factor"], sc["ratio"], "o-", color=c)
    axs[0].set(xlabel="PARSEC activation criterion", ylabel="median parcel / observed Nd", yscale="log",
               xticks=[1, 2, 3, 4], xticklabels=["N1\nr > 1 um", "N2\nSc < Smax", "N3\nr > rcrit", "N4\n> smallest act."])
    axs[1].set(xlabel="PARSEC activation criterion", ylabel="r(log Nd)", xticks=[1, 2, 3, 4],
               xticklabels=["N1", "N2", "N3", "N4"])
    axs[2].set(xlabel=f"sigma_w scaling factor (criterion N{a.criterion})", ylabel="median parcel / observed Nd",
               xscale="log", yscale="log")
    axs[2].set_xticks(a.factors)
    axs[2].set_xticklabels([f"{f:g}" for f in a.factors])
    for ax in (axs[0], axs[2]):
        ax.axhline(1, color="k", lw=0.8)
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Parcel sensitivity to the activation criterion and the updraft ({a.weighting} weighting)"
                 + (f", {a.tag} hours" if a.tag else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sensitivity.png"), dpi=150)
    print(f"-> {os.path.relpath(out, HERE)}/summary.csv, sensitivity.png")


if __name__ == "__main__":
    main()
