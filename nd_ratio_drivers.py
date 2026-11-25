#!/usr/bin/env python
"""What makes the parcel model overshoot? Parcel/observed Nd against drizzle, LWP, cloud depth and coupling.

Built for the ENA test: if drizzle (collision-coalescence) removes droplets that the adiabatic, non-precipitating
parcel keeps, the ratio parcel/observed should rise with the drizzle fraction below cloud base and with LWP, and
non-drizzling hours should sit near the ratio of the in-cloud stations (1.5-2).

Inputs: results/nd_comparison_hours.csv (observed-updraft variant) and results/cloudnet_quality/<site>_flags.csv
(cloudnet_quality.py flags: f_drizzle, f_single, f_ice from Cloudnet categorize; lwp_gm2, depth_built_m from the
built file). For ENA the categorize files come from Cloudnet's arm-graciosa site:
  python updraft_download.py --product categorize --sites ena
  python cloudnet_quality.py flags --sites ena
Per site and driver: median ratio (IQR), n and r(log Nd) in bins, the Spearman correlation of log ratio with the
driver, and nd_ratio_drivers_<site>.png (ratio against each driver, bin medians on top).

python nd_ratio_drivers.py --sites ena
python nd_ratio_drivers.py --sites ena sgp lasic epcape_pier --weighting pdf
python nd_ratio_drivers.py --sites ena --query "depth < 800 and abs(cbh_minus_lcl) <= 300" --tag _shallow_coupled
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "nd_ratio_drivers")
DRIVERS = {  # column: (label, bin edges or None for quartiles)
    "f_drizzle": ("drizzle fraction below base", [-0.001, 0.01, 0.1, 0.5, 1.0]),
    "lwp_gm2": ("LWP (g m$^{-2}$)", None),
    "depth": ("liquid depth (m)", None),
    "cbh_minus_lcl": ("cloud base - LCL (m)", [-5000, -300, 300, 1000, 10000]),
    "visibility_m": ("station visibility (m)", [0, 100, 200, 500, 1e6]),
    "rain_mmh": ("station rain (mm h$^{-1}$)", [-0.001, 0.0, 0.1, 1.0, 1e3]),
    "Nd_obs": ("observed Nd (cm$^{-3}$)", None),
}


def stats(g):
    if len(g) == 0:
        return dict(n=0, ratio=np.nan, q25=np.nan, q75=np.nan, r=np.nan, slope=np.nan)
    lr = np.log10(g["Nd_parcel"] / g["Nd_obs"])
    r = np.corrcoef(np.log10(g["Nd_obs"]), np.log10(g["Nd_parcel"]))[0, 1] if len(g) > 2 else np.nan
    sl = np.polyfit(np.log10(g["Nd_obs"]), np.log10(g["Nd_parcel"]), 1)[0] if len(g) > 2 else np.nan
    return dict(n=len(g), ratio=10 ** lr.median(), q25=10 ** lr.quantile(.25), q75=10 ** lr.quantile(.75), r=r, slope=sl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=["ena"])
    ap.add_argument("--weighting", choices=["pdf", "flux"], default="flux")
    ap.add_argument("--query", help="pandas query to subset hours first, e.g. "
                                    "\"depth < 800 and abs(cbh_minus_lcl) <= 300\"")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(os.path.join(HERE, "results", "nd_comparison_hours.csv"), parse_dates=["hour"])
    t = t[(t["variant"] == "obs") & (t["weighting"] == a.weighting) & t["usable"]]
    t = t[(t["Nd_obs"] > 0) & (t["Nd_parcel"] > 0)]
    rows = []
    for site in a.sites:
        d = t[t["site"] == site].set_index("hour")
        fp = os.path.join(HERE, "results", "cloudnet_quality", f"{site}_flags.csv")
        if os.path.exists(fp):
            d = d.join(pd.read_csv(fp, parse_dates=["hour"]).set_index("hour"), how="left", rsuffix="_flag")
        else:
            print(f"{site}: no flags file (run cloudnet_quality.py flags --sites {site}); built-file drivers only")
        if "liquid_depth_cat_m" in d or "depth_built_m" in d:
            d["depth"] = d.get("liquid_depth_cat_m", pd.Series(np.nan, index=d.index)).fillna(
                d.get("depth_built_m", pd.Series(np.nan, index=d.index)))
        if a.query:
            n0 = len(d)
            d = d.query(a.query)
            print(f"\n{site}: query '{a.query}' keeps {len(d)} of {n0} hours")
        print(f"\n{site}: {len(d)} hours, all: " + "ratio {ratio:.2f} ({q25:.2f}-{q75:.2f}), r {r:.2f}, slope {slope:.2f}".format(**stats(d)))
        drv = [k for k in DRIVERS if k in d and d[k].notna().sum() >= 20]
        if not drv:
            print("  (fewer than 20 hours with any driver: no bins or figure)")
            continue
        fig, axs = plt.subplots(1, len(drv), figsize=(3.8 * len(drv), 3.6), squeeze=False)
        for ax, k in zip(axs[0], drv):
            lab, edges = DRIVERS[k]
            x = d[k].astype(float)
            ok = x.notna()
            lr = np.log10(d["Nd_parcel"] / d["Nd_obs"])
            rho, p = spearmanr(x[ok], lr[ok])
            bins = pd.cut(x, edges) if edges else pd.qcut(x, 4, duplicates="drop")
            print(f"  {lab} (Spearman rho with log ratio {rho:+.2f}, p {p:.2g}, n {ok.sum()}):")
            for b, g in d[ok].groupby(bins[ok], observed=True):
                s = stats(g)
                rows.append(dict(site=site, driver=k, bin=str(b), **s))
                print(f"     {str(b):24s} n {s['n']:4d}  ratio {s['ratio']:.2f} ({s['q25']:.2f}-{s['q75']:.2f})  r {s['r']:.2f}")
            ax.scatter(x[ok], 10 ** lr[ok], s=6, alpha=0.4, color="0.4", lw=0)
            med = d[ok].groupby(bins[ok], observed=True).apply(
                lambda g: pd.Series({"x": g[k].median(), "y": 10 ** np.log10(g["Nd_parcel"] / g["Nd_obs"]).median()}))
            ax.plot(med["x"], med["y"], "o-", color="#d62728", ms=6, label="bin median")
            ax.axhline(1, color="k", lw=0.8)
            ax.set(yscale="log", xlabel=lab, title=f"rho = {rho:+.2f}")
            if k in ("lwp_gm2", "Nd_obs"):
                ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axs[0, 0].set_ylabel(f"parcel / observed Nd ({a.weighting})")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"{site}: what the parcel/observed Nd ratio depends on")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"nd_ratio_drivers_{site}{a.tag}.png"), dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"bins{a.tag}.csv"), index=False, float_format="%.3g")
    print(f"-> {os.path.relpath(OUT, HERE)}/bins.csv, nd_ratio_drivers_<site>.png")


if __name__ == "__main__":
    main()
