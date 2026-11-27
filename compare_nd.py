#!/usr/bin/env python
"""Updraft-weight every site's PARSEC runs and compare modelled with observed Nd.

1. For each site with runs/<site>/chunk_*.csv, weight_updrafts.py is run with updrafts/<site>_w_cloudbase.nc,
   once with "pdf" and once with "flux" weighting. Extra variants:
     zeppelin       the station updraft with its observed mean, and with mean 0 (sigma_w only)
     puijo, pallas  fixed sigma_w at the 10th and 90th percentiles of the station climatology from
                    paper_updrafts.py (else the boreal lidar sites, BOUNDS)
2. All results/<site>_parcel_nd_*.nc are collected into results/nd_comparison_hours.csv (one row per site,
   variant and hour) and summarised in results/nd_comparison_summary.csv: hours, median parcel/observed ratio
   and its interquartile range, and the correlation of log Nd, per site and variant, and per composition tier
   and cloud-base T/P source.
3. Zeppelin: the parcel activation diameter (2 x PARSEC's smallest activated dry radius, weighted over the
   hour's updraft PDF like Nd) against the observed integral activation diameter (results/zeppelin_activation.nc).
4. results/nd_parcel_vs_obs.png: parcel against observed Nd, one panel per weighting.

Hours with placeholder T0/P0 are left out of the summary (kept in the hours file, flagged). Every hour is also
classed by coupling (in-cloud station; coupled: |cloud base - surface LCL| <= 300 m; decoupled; unknown where there
is no surface met) and by cloud-base temperature T0 (warm >= 0 C; supercooled -10..0 C; cold < -10 C), with
summaries over all sites and per site (flux weighting).

Examples
--------
python compare_nd.py
python compare_nd.py --sites ena sgp --skip-weighting
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
RES = os.path.join(HERE, "results")
BOUNDS = {"puijo": (0.23, 0.70), "pallas": (0.23, 0.70)}
COUPLED_M = 300.0                                   # |cloud base - surface LCL| for "coupled", as in build_thermo.py
TEMP_EDGES = (263.15, 273.15)                       # cold < -10 C <= supercooled < 0 C <= warm
AF_OBS_MAX = 2.0                                    # screen: observed Nd / N100 above this is not physical


def classes(site, hours):
    """Per hour: coupling class (in-cloud station / coupled / decoupled / unknown) and cloud-base temperature
    class (warm / supercooled / cold), from inputs/<site>_parcel_inputs.nc (cbh_minus_lcl, T0)."""
    sys.path.insert(0, HERE)
    from sites import SITES
    p = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    out = pd.DataFrame({"coupling": "unknown", "temp": "unknown", "T0": np.nan, "cbh_minus_lcl": np.nan},
                       index=pd.DatetimeIndex(hours))
    if not os.path.exists(p):
        return out
    with xr.open_dataset(p) as d:
        h = pd.DatetimeIndex(d["hour"].values)
        T0 = pd.Series(d["T0"].values.astype(float), index=h)
        dz = pd.Series(d["cbh_minus_lcl"].values.astype(float) if "cbh_minus_lcl" in d else np.nan, index=h)
        src = pd.Series(d["thermo_source"].values.astype(str), index=h)
    T0, dz, src = T0.reindex(out.index), dz.reindex(out.index), src.reindex(out.index)
    out["T0"], out["cbh_minus_lcl"] = T0.values, dz.values
    if SITES[site]["kind"] == "insitu":
        out["coupling"] = "in-cloud station"
    elif SITES[site]["kind"] == "aircraft":
        out["coupling"] = "aircraft (measured base)"
    else:
        out.loc[dz.abs() <= COUPLED_M, "coupling"] = "coupled"
        out.loc[dz.abs() > COUPLED_M, "coupling"] = "decoupled"
    T0 = T0.where(src != "placeholder")
    out.loc[T0 >= TEMP_EDGES[1], "temp"] = "warm (>= 0 C)"
    out.loc[(T0 >= TEMP_EDGES[0]) & (T0 < TEMP_EDGES[1]), "temp"] = "supercooled (-10..0 C)"
    out.loc[T0 < TEMP_EDGES[0], "temp"] = "cold (< -10 C)"
    return out


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
    bounds = BOUNDS.get(site, ())
    if site in BOUNDS and os.path.exists(wf):            # paper_updrafts.py stores the station's own 10/90 %
        with xr.open_dataset(wf) as d:
            if "sigma_w_p10" in d.attrs:
                bounds = (round(float(d.attrs["sigma_w_p10"]), 2), round(float(d.attrs["sigma_w_p90"]), 2))
    for s in bounds:
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
            for v, col in (("Ntot_input", "Ntot"), ("sigma_w", "sigma_w"), ("Dact_parcel_nm", "Dact_nm"),
                           ("Smax_parcel", "Smax_pct")):
                df[col] = d[v].values if v in d else np.nan          # older result files lack these
        c = classes(site, df["hour"])
        df["coupling"], df["temp"] = c["coupling"].values, c["temp"].values
        df["T0"], df["cbh_minus_lcl"] = c["T0"].values, c["cbh_minus_lcl"].values
        df.insert(0, "weighting", wt)
        df.insert(0, "variant", tag)
        df.insert(0, "site", site)
        rows.append(df)
    t = pd.concat(rows, ignore_index=True)
    from sites import SITES, station_of
    t.insert(1, "station", t["site"].map(station_of))
    t.insert(2, "nd_method", t["site"].map(lambda s: SITES.get(s, {}).get("kind", "?")))
    t["tier"] = t["comp_source"].str.split("+").str[0]
    t["thermo"] = t["thermo_source"].str.replace(r"_(decoupled|cloud_base|station)$", r"_\1", regex=True)
    t["usable"] = (t["thermo_source"] != "placeholder") & np.isfinite(t["Nd_parcel"]) & (t["Nd_obs"] > 0)
    t["ratio"] = t["Nd_parcel"] / t["Nd_obs"]
    t["af_obs"] = t["Nd_obs"] / t["N100"]
    t["af_parcel"] = t["Nd_parcel"] / t["N100"]
    # physical bounds on the observed Nd: no more droplets than particles, and not far more than N(>100 nm)
    t["phys_ok"] = (t["af_obs"] <= AF_OBS_MAX) & ~(t["Nd_obs"] > t["Ntot"])
    return t


def screen_table(t):
    """Hours removed by the physical-bounds screen, per site (observed-updraft, flux)."""
    u = t[t["usable"] & (t["variant"] == "obs") & (t["weighting"] == "flux")]
    return u.groupby("site").apply(lambda g: pd.Series({
        "hours": len(g), "af_obs_gt_max": int((g["af_obs"] > AF_OBS_MAX).sum()),
        "Nd_obs_gt_Ntot": int((g["Nd_obs"] > g["Ntot"]).sum()),
        "kept": int(g["phys_ok"].sum()), "kept_frac": g["phys_ok"].mean()}), include_groups=False).reset_index()


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


def plot(t, outdir=None):
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
    os.makedirs(outdir or RES, exist_ok=True)
    out = os.path.join(outdir or RES, "nd_parcel_vs_obs.png")
    fig.savefig(out, dpi=150)
    print(f"-> {os.path.relpath(out, HERE)}")


COLOUR_BY = {
    "coupling": {"coupled": "#1f77b4", "decoupled": "#d62728", "in-cloud station": "#2ca02c", "unknown": "0.55"},
    "temp": {"warm (>= 0 C)": "#d62728", "supercooled (-10..0 C)": "#1f77b4", "cold (< -10 C)": "#9467bd"},
}


def site_scatter(t, weighting="flux", colour_by="none", outdir=None):
    """Observed vs parcel Nd per site (observed-updraft variant, usable hours): one panel per site in
    results/nd_scatter_<weighting>.png, and one file per site in results/nd_scatter/<site>_<weighting>.png.
    Each panel: 1:1 (solid) and factor-2 (dotted) lines, n, median parcel/observed ratio with IQR, Pearson r
    of log Nd, and the log-log regression slope (OLS of log parcel on log observed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    outdir = outdir or RES
    os.makedirs(outdir, exist_ok=True)
    u = t[t["usable"] & (t["variant"] == "obs") & (t["weighting"] == weighting)]
    u = u[(u["Nd_obs"] > 0) & (u["Nd_parcel"] > 0)]
    sites = sorted(u["site"].unique())
    if not sites:
        return
    cmap = COLOUR_BY.get(colour_by)

    def panel(ax, g, site):
        x, y = g["Nd_obs"].to_numpy(float), g["Nd_parcel"].to_numpy(float)
        lo = 10 ** np.floor(np.log10(min(x.min(), y.min())))
        hi = 10 ** np.ceil(np.log10(max(x.max(), y.max())))
        a = 0.6 if len(g) < 300 else 0.35 if len(g) < 2000 else 0.2
        if cmap:
            for k, c in cmap.items():
                m = (g[colour_by] == k).to_numpy()
                if m.any():
                    ax.scatter(x[m], y[m], s=6, color=c, alpha=a, lw=0, rasterized=True)
        else:
            ax.scatter(x, y, s=6, color="#1f77b4", alpha=a, lw=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        for f in (0.5, 2):
            ax.plot([lo, hi], [lo * f, hi * f], "k:", lw=0.6)
        lx, ly = np.log10(x), np.log10(y)
        r = np.corrcoef(lx, ly)[0, 1] if len(g) > 2 else np.nan
        slope = np.polyfit(lx, ly, 1)[0] if len(g) > 2 else np.nan
        q = np.log10(y / x)
        ax.text(0.03, 0.97, f"n = {len(g)}\nratio {10 ** np.median(q):.2f} ({10 ** np.percentile(q, 25):.2f}-"
                f"{10 ** np.percentile(q, 75):.2f})\nr(log) = {r:.2f}, slope {slope:.2f}",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), title=site)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, which="major")

    handles = [Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in cmap.items()] if cmap else []
    note = f"; colour: {colour_by}" if cmap else ""
    nc = 5
    nr = int(np.ceil(len(sites) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.3 * nr), squeeze=False)
    for ax, site in zip(axs.flat, sites):
        panel(ax, u[u["site"] == site], site)
    for ax in axs.flat[len(sites):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("parcel Nd (cm$^{-3}$)")
    for ax in axs[-1]:
        ax.set_xlabel("observed Nd (cm$^{-3}$)")
    if handles:
        fig.legend(handles=handles, loc="lower right", fontsize=8, markerscale=1.5)
    fig.suptitle(f"Parcel vs observed Nd per site, {weighting} weighting (1:1 solid, factor 2 dotted{note})")
    fig.tight_layout()
    path = os.path.join(outdir, f"nd_scatter_{weighting}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    d = os.path.join(outdir, "nd_scatter")
    os.makedirs(d, exist_ok=True)
    for site in sites:
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        panel(ax, u[u["site"] == site], f"{site}, {weighting} weighting")
        ax.set(xlabel="observed Nd (cm$^{-3}$)", ylabel="parcel Nd (cm$^{-3}$)")
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"{site}_{weighting}.png"), dpi=150)
        plt.close(fig)
    print(f"-> {os.path.relpath(path, HERE)}, {os.path.relpath(d, HERE)}/<site>_{weighting}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--rundir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--skip-weighting", action="store_true", help="use the result files already there")
    ap.add_argument("--scatter-only", action="store_true",
                    help="only redraw the per-site scatter plots from results/nd_comparison_hours.csv")
    ap.add_argument("--colour-by", choices=["none", "coupling", "temp"], default="none",
                    help="colour the per-site scatter points by coupling or cloud-base temperature class")
    ap.add_argument("--screen", action="store_true",
                    help=f"summaries and plots only from hours passing the physical-bounds screen "
                         f"(Nd_obs/N100 <= {AF_OBS_MAX:g} and Nd_obs <= Ntot); the hourly CSV keeps all hours with phys_ok")
    a = ap.parse_args()
    if a.scatter_only:
        t = pd.read_csv(os.path.join(RES, "nd_comparison_hours.csv"))
        if a.sites:
            t = t[t["site"].isin(a.sites)]
        if a.screen and "phys_ok" in t:
            t = t[t["phys_ok"]]
        for wt in ("flux", "pdf"):
            site_scatter(t, wt, a.colour_by, outdir=os.path.join(RES, "screened") if a.screen else None)
        return
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
    pd.set_option("display.width", 200)
    print("\nPhysical-bounds screen (observed-updraft, flux; Ntot = parcel input total):")
    print(screen_table(t).to_string(index=False, float_format="{:.2f}".format))
    suffix = ""
    if a.screen:
        t = t[t["phys_ok"]].copy()
        suffix = "_screened"
        print("summaries below use only hours passing the screen")
    by_site = summarise(t, ["site", "variant", "weighting"])
    by_tier = summarise(t[t["variant"] == "obs"], ["weighting", "tier"])
    by_thermo = summarise(t[t["variant"] == "obs"], ["weighting", "thermo"])
    ob = t[t["variant"] == "obs"]
    by_coup = summarise(ob, ["weighting", "coupling"])
    by_temp = summarise(ob, ["weighting", "temp"])
    by_site_coup = summarise(ob[ob["weighting"] == "flux"], ["site", "coupling"])
    by_site_temp = summarise(ob[ob["weighting"] == "flux"], ["site", "temp"])
    multi = ob.groupby("station")["site"].transform("nunique") > 1          # stations with more than one site
    by_station = summarise(ob, ["station", "weighting"])
    by_station_method = summarise(ob[multi & (ob["weighting"] == "flux")], ["station", "nd_method", "site"])
    with open(os.path.join(RES, f"nd_comparison_summary{suffix}.csv"), "w") as fh:
        for name, d in (("by_site", by_site), ("by_composition_tier", by_tier), ("by_thermo_source", by_thermo),
                        ("by_coupling", by_coup), ("by_temperature", by_temp),
                        ("by_site_and_coupling_flux", by_site_coup), ("by_site_and_temperature_flux", by_site_temp),
                        ("by_station", by_station), ("by_station_and_nd_method_flux", by_station_method)):
            fh.write(f"# {name}\n")
            d.to_csv(fh, index=False, float_format="%.3g")
    fmt = {c: "{:.2f}".format for c in ("ratio_median", "ratio_q25", "ratio_q75", "r_log")}
    for name, d in (("By site", by_site), ("By composition tier", by_tier), ("By cloud-base T/P source", by_thermo),
                    ("By coupling (|cloud base - LCL| <= 300 m)", by_coup), ("By cloud-base temperature", by_temp),
                    ("By site and coupling, flux weighting", by_site_coup),
                    ("By site and cloud-base temperature, flux weighting", by_site_temp),
                    ("By station (sites with the same aerosol pooled; observed-updraft weighting)", by_station),
                    ("Stations with two Nd methods, flux weighting", by_station_method)):
        print(f"\n{name} (parcel / observed Nd):")
        print(d.to_string(index=False, formatters=fmt, float_format="{:.0f}".format))
    zeppelin_activation(a)
    figdir = os.path.join(RES, "screened") if a.screen else RES
    plot(t, figdir)
    for wt in ("flux", "pdf"):
        site_scatter(t, wt, a.colour_by, outdir=figdir)
    print(f"-> results/nd_comparison_hours.csv, results/nd_comparison_summary{suffix}.csv")


if __name__ == "__main__":
    main()
