#!/usr/bin/env python
"""Updraft-weight every site's PARSEC runs and compare modelled with observed Nd.

1. For each site with runs/<site>/chunk_*.csv, weight_updrafts.py is run with updrafts/<site>_w_cloudbase.nc,
   once with "pdf" and once with "flux" weighting. Extra variants:
     zeppelin       the station updraft with its observed mean, and with mean 0 (sigma_w only)
     puijo, pallas  fixed sigma_w at the 10th and 90th percentiles of the boreal lidar sites (bounds)
2. All results/<site>_parcel_nd_*.nc are collected into results/nd_comparison_hours.csv (one row per site,
   variant and hour) and summarised in results/nd_comparison_summary.csv: hours, median parcel/observed ratio
   and its interquartile range, and the correlation of log Nd, per site and variant, and per composition tier
   and cloud-base T/P source.
3. Zeppelin: the parcel activation diameter (2 x PARSEC's smallest activated dry radius, weighted over the
   hour's updraft PDF like Nd) against the observed integral activation diameter (results/zeppelin_activation.nc).
4. results/nd_parcel_vs_obs.png: parcel against observed Nd, one panel per weighting.

Hours with placeholder T0/P0 are left out of the summary (kept in the hours file, flagged).

Examples
--------
python compare_nd.py
python compare_nd.py --sites ena sgp --skip-weighting
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BOUNDS = {"puijo": (0.23, 0.70), "pallas": (0.23, 0.70)}


def run_weighting(site, a):
    """Call weight_updrafts.py for the site's variants; returns the output files."""
    outs = []
    wf = os.path.join(HERE, "updrafts", f"{site}_w_cloudbase.nc")
    variants = []
    if os.path.exists(wf):
        variants.append(("obs", ["--w-file", wf]))
    if site == "zeppelin" and os.path.exists(wf):
        z = os.path.join(HERE, "updrafts", "zeppelin_w_cloudbase_zeromean.nc")
        with xr.open_dataset(wf) as d:
            d = d.load()
        d["w_mean"] = d["w_mean"] * 0.0
        d = d.drop_vars([v for v in ("w",) if v in d])
        d.attrs["w_source"] = "zeppelin_station_5min_zero_mean"
        d.to_netcdf(z)
        variants.append(("zeromean", ["--w-file", z]))
    for s in BOUNDS.get(site, ()):
        variants.append((f"sigma{s:g}", ["--sigma-w", str(s)]))
    for tag, args in variants:
        for wt in ("pdf", "flux"):
            out = os.path.join(RES, f"{site}_parcel_nd_{tag}_{wt}.nc")
            cmd = [sys.executable, os.path.join(HERE, "weight_updrafts.py"), "--site", site, "--rundir", a.rundir,
                   "--weighting", wt, "--out", out] + args
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {site} {tag} {wt}: failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:150]}")
                continue
            outs.append((tag, wt, out))
    return outs


def collect(files):
    rows = []
    for site, tag, wt, f in files:
        with xr.open_dataset(f) as d:
            df = pd.DataFrame({"hour": pd.DatetimeIndex(d["hour"].values), "Nd_parcel": d["Nd_parcel"].values,
                               "Nd_obs": d["Nd_obs"].values, "N100": d["N100"].values,
                               "comp_source": d["comp_source"].values.astype(str),
                               "thermo_source": d["thermo_source"].values.astype(str),
                               "w_source": d["w_source"].values.astype(str),
                               "frac_w_outside": d["frac_w_outside"].values})
        df.insert(0, "weighting", wt)
        df.insert(0, "variant", tag)
        df.insert(0, "site", site)
        rows.append(df)
    t = pd.concat(rows, ignore_index=True)
    t["tier"] = t["comp_source"].str.split("+").str[0]
    t["thermo"] = t["thermo_source"].str.replace(r"_(decoupled|cloud_base|station)$", r"_\1", regex=True)
    t["usable"] = (t["thermo_source"] != "placeholder") & np.isfinite(t["Nd_parcel"]) & (t["Nd_obs"] > 0)
    t["ratio"] = t["Nd_parcel"] / t["Nd_obs"]
    return t


def summarise(t, by):
    u = t[t["usable"]]

    def f(g):
        lr = np.log10(g["ratio"])
        r = np.corrcoef(np.log10(g["Nd_parcel"]), np.log10(g["Nd_obs"]))[0, 1] if len(g) > 2 else np.nan
        return pd.Series({"hours": len(g), "Nd_obs_median": g["Nd_obs"].median(),
                          "Nd_parcel_median": g["Nd_parcel"].median(),
                          "ratio_median": 10 ** lr.median(), "ratio_q25": 10 ** lr.quantile(.25),
                          "ratio_q75": 10 ** lr.quantile(.75), "r_log": r})
    return u.groupby(by).apply(f, include_groups=False).reset_index()


def zeppelin_activation(a):
    act = os.path.join(RES, "zeppelin_activation.nc")
    files = sorted(glob.glob(os.path.join(a.rundir, "zeppelin", "chunk_*.csv")))
    if not (os.path.exists(act) and files):
        return
    from weight_updrafts import gaussian_positive
    df = pd.concat([pd.read_csv(f, parse_dates=["hour"]) for f in files], ignore_index=True)
    df = df[df["status"] == "ok"]
    r = df.pivot_table(index="hour", columns="w", values="r_act_min_nm")
    wgrid = r.columns.to_numpy(float)
    with xr.open_dataset(act) as d:
        obs = pd.Series(d["d_act_int"].values, index=pd.DatetimeIndex(d["hour"].values))
    wf = xr.open_dataset(os.path.join(HERE, "updrafts", "zeppelin_w_cloudbase.nc"))
    sig = pd.Series(wf["sigma_w"].values, index=pd.DatetimeIndex(wf["hour"].values))
    mean = pd.Series(wf["w_mean"].values, index=sig.index)
    print("\nZeppelin activation diameter, parcel vs observed (integral method):")
    for tag, m in (("station mean w", mean), ("mean 0", mean * 0)):
        out = []
        for h in r.index:
            s, mu, o = sig.get(h, np.nan), m.get(h, np.nan), obs.get(h, np.nan)
            row = r.loc[h].to_numpy(float)
            ok = np.isfinite(row) & (row > 0)
            if not (np.isfinite(s) and s > 0 and np.isfinite(o) and ok.sum() >= 3):
                continue
            wq, pq = gaussian_positive(s, mu)
            if wq is None:
                continue
            dq = 2 * np.exp(np.interp(np.log(np.clip(wq, wgrid[ok][0], wgrid[ok][-1])), np.log(wgrid[ok]), np.log(row[ok])))
            out.append((np.sum(dq * pq) / np.sum(pq), o))
        if out:
            x = np.array(out)
            q = x[:, 0] / x[:, 1]
            print(f"   {tag:15s} {len(x)} hours: parcel {np.median(x[:, 0]):.0f} nm vs observed {np.median(x[:, 1]):.0f} nm, "
                  f"median ratio {np.median(q):.2f} (IQR {np.percentile(q, 25):.2f}-{np.percentile(q, 75):.2f})")


def plot(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = t[t["usable"] & (t["variant"] == "obs")]
    sites = sorted(u["site"].unique())
    cmap = plt.get_cmap("tab20")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, wt in zip(axs, ("pdf", "flux")):
        for k, s in enumerate(sites):
            g = u[(u["weighting"] == wt) & (u["site"] == s)]
            ax.scatter(g["Nd_obs"], g["Nd_parcel"], s=4, alpha=0.4, color=cmap(k % 20), label=s, rasterized=True)
        lim = [1, 3000]
        ax.plot(lim, lim, "k-", lw=0.8)
        for f in (0.5, 2):
            ax.plot(lim, [l * f for l in lim], "k:", lw=0.6)
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="observed Nd (cm$^{-3}$)",
               title=f"{wt} weighting")
    axs[0].set_ylabel("parcel Nd (cm$^{-3}$)")
    axs[1].legend(fontsize=7, markerscale=3, ncol=2, loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "nd_parcel_vs_obs.png")
    fig.savefig(out, dpi=150)
    print(f"-> {os.path.relpath(out, HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--rundir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--skip-weighting", action="store_true", help="use the result files already there")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    sites = a.sites or sorted(d for d in os.listdir(a.rundir) if glob.glob(os.path.join(a.rundir, d, "chunk_*.csv")))
    files = []
    for s in sites:
        if a.skip_weighting:
            for f in sorted(glob.glob(os.path.join(RES, f"{s}_parcel_nd_*_*.nc"))):
                tag, wt = os.path.basename(f)[len(s) + len("_parcel_nd_"):-3].rsplit("_", 1)
                files.append((s, tag, wt, f))
        else:
            print(f"{s}: weighting", flush=True)
            files += [(s, tag, wt, f) for tag, wt, f in run_weighting(s, a)]
    if not files:
        raise SystemExit("no results to compare")
    t = collect(files)
    t.to_csv(os.path.join(RES, "nd_comparison_hours.csv"), index=False)
    by_site = summarise(t, ["site", "variant", "weighting"])
    by_tier = summarise(t[t["variant"] == "obs"], ["weighting", "tier"])
    by_thermo = summarise(t[t["variant"] == "obs"], ["weighting", "thermo"])
    with open(os.path.join(RES, "nd_comparison_summary.csv"), "w") as fh:
        for name, d in (("by_site", by_site), ("by_composition_tier", by_tier), ("by_thermo_source", by_thermo)):
            fh.write(f"# {name}\n")
            d.to_csv(fh, index=False, float_format="%.3g")
    pd.set_option("display.width", 200)
    fmt = {c: "{:.2f}".format for c in ("ratio_median", "ratio_q25", "ratio_q75", "r_log")}
    for name, d in (("By site", by_site), ("By composition tier", by_tier), ("By cloud-base T/P source", by_thermo)):
        print(f"\n{name} (parcel / observed Nd):")
        print(d.to_string(index=False, formatters=fmt, float_format="{:.0f}".format))
    zeppelin_activation(a)
    plot(t)
    print("-> results/nd_comparison_hours.csv, results/nd_comparison_summary.csv")


if __name__ == "__main__":
    main()
