#!/usr/bin/env python
"""Does parcel-observation agreement at the remote-sensing sites improve in clouds that suit the retrieval?

Step 1 (flags): per cloud hour, from the Cloudnet categorize files already downloaded for the updrafts
(data/updraft/cloudnet_categorize/<cloudnet id>/, see updraft_download.py --product categorize), over the
profiles with liquid (category bit 0):
  f_single   fraction with one liquid layer (liquid runs separated by more than --gap-gates gates count apart)
  f_drizzle  fraction with falling, not-cold hydrometeors (bit 1 set, bit 2 clear) below the liquid base
  f_ice      fraction with falling cold hydrometeors (bits 1 and 2) anywhere below 6 km (ice / mixed phase)
  f_rain     fraction of all profiles with rain at the surface (rain_detected / is_rain)
written to results/cloudnet_quality/<site>_flags.csv. From the observational built file ($OBS_REPO): LWP
(lwp_meas, g m-2), within-hour Nd spread (nd_iqr_ratio), liquid depth; coupling from compare_nd.classes.

Step 2 (screen): the hourly parcel and observed Nd (results/nd_comparison_hours.csv, observed-updraft variant)
are screened cumulatively
  all -> single layer -> no drizzle -> no ice -> no rain -> LWP in --lwp-range -> homogeneous (nd_iqr_ratio
  <= --max-iqr) [-> coupled, with --require-coupled]
and n, median parcel/observed ratio, r(log Nd) and log-log slope are reported after each step, per site.
Hours whose categorize flags are missing fail the categorize steps (counted). ARM sites without Cloudnet
categorize files (the ARM NDROP sites, *_cloudnet ARM variants) get only the built-file steps.
Writes results/cloudnet_quality/steps.csv, steps.png (r and ratio per step), and ideal_hours.csv (site, hour)
for susceptibility_parcel.py --hours-file results/cloudnet_quality/ideal_hours.csv --tag ideal.

python cloudnet_quality.py flags                     # slow: reads the categorize files (once)
python cloudnet_quality.py screen                    # fast; options change the thresholds
python cloudnet_quality.py screen --require-coupled --lwp-range 40 250
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sites import SITES  # noqa: E402
from updraft_download import CLOUDNET_RADAR_SITES  # noqa: E402

OUT = os.path.join(HERE, "results", "cloudnet_quality")
CAT = os.path.join(HERE, "data", "updraft", "cloudnet_categorize")


def n_runs(mask, gap):
    """Number of runs of True along the last axis; runs separated by <= gap False gates count as one."""
    m = mask.astype(np.int32)
    cs = np.concatenate([np.zeros((m.shape[0], 1), np.int32), np.cumsum(m, axis=1)], axis=1)
    j = np.arange(m.shape[1])
    prev = cs[:, j] - cs[:, np.clip(j - gap - 1, 0, None)]        # liquid gates among the previous gap+1
    return (mask & (prev == 0)).sum(axis=1)


def flags_site(site, hours, a):
    cid = CLOUDNET_RADAR_SITES.get(site)
    files = sorted(glob.glob(os.path.join(CAT, cid or "-", "*.nc")))
    if not files:
        return None
    days = set(hours.normalize())
    rows = []
    for f in files:
        try:
            day = pd.Timestamp(os.path.basename(f)[:8])
        except ValueError:
            continue
        if day not in days:
            continue
        try:
            with xr.open_dataset(f) as ds:
                t = pd.DatetimeIndex(ds["time"].values)
                hr = t.floor("h")
                keep = np.asarray(hr.isin(hours))
                if not keep.any():
                    continue
                alt = float(np.nanmedian(ds["altitude"].values))
                z = ds["height"].values.astype(float) - alt
                cat = ds["category_bits"].values[keep].astype(np.int64)
                rain = None
                for rv in ("rain_detected", "is_rain"):
                    if rv in ds:
                        rain = ds[rv].values[keep].astype(float) > 0
                        break
        except Exception as ex:                                                      # noqa: BLE001
            print(f"    skip {os.path.basename(f)}: {str(ex)[:80]}")
            continue
        liq = (cat & 1) > 0
        fall = (cat & 2) > 0
        cold = (cat & 4) > 0
        has = liq.any(axis=1)
        base_i = np.where(has, np.argmax(liq, axis=1), len(z))
        below = np.arange(len(z))[None, :] < base_i[:, None]
        low = (z >= 0) & (z <= 6000)
        drizzle = (fall & ~cold & below & low[None, :]).any(axis=1)
        ice = (fall & cold & low[None, :]).any(axis=1)
        single = n_runs(liq & low[None, :], a.gap_gates) == 1
        top_i = len(z) - 1 - np.argmax(liq[:, ::-1], axis=1)
        depth = np.where(has, z[np.clip(top_i, 0, len(z) - 1)] - z[np.clip(base_i, 0, len(z) - 1)], np.nan)
        rows.append(pd.DataFrame(dict(hour=hr[keep], liquid=has, single=single & has, drizzle=drizzle & has,
                                      ice=ice & has, rain=rain if rain is not None else np.nan, depth=depth)))
    if not rows:
        return None
    df = pd.concat(rows)
    g = df.groupby("hour")
    nl = g["liquid"].sum().replace(0, np.nan)
    out = pd.DataFrame({"n_profiles": g.size(), "n_liquid": g["liquid"].sum(),
                        "f_single": g["single"].sum() / nl, "f_drizzle": g["drizzle"].sum() / nl,
                        "f_ice": g["ice"].sum() / nl, "f_rain": g["rain"].mean(),
                        "liquid_depth_cat_m": g["depth"].median()})
    return out.reindex(hours)


def built_vars(site, hours, obs_repo):
    """LWP (g m-2), nd_iqr_ratio and liquid depth from the observational built file."""
    out = pd.DataFrame(index=hours, columns=["lwp_gm2", "nd_iqr_ratio", "depth_built_m"], dtype=float)
    if not obs_repo or site not in SITES:
        return out
    p = os.path.join(obs_repo, SITES[site]["built"])
    if not os.path.exists(p):
        return out
    with xr.open_dataset(p) as ds:
        t = pd.DatetimeIndex(ds["time"].values).floor("h")
        d = pd.DataFrame(index=t)
        for v, k, f in (("lwp_meas", "lwp_gm2", 1000.0), ("lwp", "lwp_gm2", 1.0), ("nd_iqr_ratio", "nd_iqr_ratio", 1.0)):
            if v in ds and k not in d:
                x = ds[v].values.astype(float)
                if v == "lwp" and np.nanmedian(x) < 5:       # kg m-2
                    x = x * 1000.0
                d[k] = x * f
        if "cloud_top_height" in ds and "cloud_base_height" in ds:
            d["depth_built_m"] = ds["cloud_top_height"].values.astype(float) - ds["cloud_base_height"].values.astype(float)
    d = d[~d.index.duplicated()]
    return d.reindex(hours).reindex(columns=out.columns)


def flags(a):
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(os.path.join(HERE, "results", "nd_comparison_hours.csv"), parse_dates=["hour"])
    t = t[t["variant"] == "obs"]
    sites = a.sites or sorted(s for s in t["site"].unique() if SITES.get(s, {}).get("kind") in ("cloudnet", "ndrop"))
    for site in sites:
        hours = pd.DatetimeIndex(sorted(t.loc[t["site"] == site, "hour"].unique()))
        f = flags_site(site, hours, a)
        b = built_vars(site, hours, a.obs_repo)
        d = b if f is None else f.join(b)
        d.index.name = "hour"
        d.to_csv(os.path.join(OUT, f"{site}_flags.csv"), float_format="%.4g")
        msg = "no categorize files" if f is None else (f"categorize for {f['n_profiles'].notna().sum()} of {len(hours)} h; "
              f"median f_single {f['f_single'].median():.2f}, f_drizzle {f['f_drizzle'].median():.2f}, "
              f"f_ice {f['f_ice'].median():.2f}")
        print(f"{site:16s} {msg}; LWP median {b['lwp_gm2'].median():.0f} g m-2", flush=True)


def fitstats(g):
    x, y = np.log10(g["Nd_obs"].values), np.log10(g["Nd_parcel"].values)
    if len(g) < 3:
        return dict(n=len(g), ratio=np.nan, r_log=np.nan, slope=np.nan)
    return dict(n=len(g), ratio=10 ** np.median(y - x), r_log=np.corrcoef(x, y)[0, 1], slope=np.polyfit(x, y, 1)[0])


def screen(a):
    t = pd.read_csv(os.path.join(HERE, "results", "nd_comparison_hours.csv"), parse_dates=["hour"])
    t = t[(t["variant"] == "obs") & (t["weighting"] == a.weighting) & t["usable"]]
    t = t[(t["Nd_obs"] > 0) & (t["Nd_parcel"] > 0)]
    rows, ideal = [], []
    for fp in sorted(glob.glob(os.path.join(OUT, "*_flags.csv"))):
        site = os.path.basename(fp)[:-len("_flags.csv")]
        if a.sites and site not in a.sites:
            continue
        fl = pd.read_csv(fp, parse_dates=["hour"]).set_index("hour")
        d = t[t["site"] == site].set_index("hour").join(fl, how="left")
        has_cat = "f_single" in d and d["f_single"].notna().any()
        steps = [("all", np.ones(len(d), bool))]
        if has_cat:
            steps += [("single layer", (d["f_single"] >= a.min_single).values),
                      ("no drizzle", (d["f_drizzle"] <= a.max_drizzle).values),
                      ("no ice", (d["f_ice"] <= a.max_ice).values),
                      ("no rain", ~(d["f_rain"] > a.max_rain).values & d["f_rain"].notna().values
                       if d["f_rain"].notna().any() else np.ones(len(d), bool))]
        lwp = d["lwp_gm2"] if "lwp_gm2" in d else pd.Series(np.nan, index=d.index)
        if lwp.notna().any():
            steps.append(("LWP range", lwp.between(*a.lwp_range).values))
        iqr = d["nd_iqr_ratio"] if "nd_iqr_ratio" in d else pd.Series(np.nan, index=d.index)
        if iqr.notna().any():
            steps.append(("homogeneous", (iqr <= a.max_iqr).values))
        if a.require_coupled:
            steps.append(("coupled", (d["coupling"] == "coupled").values))
        keep = np.ones(len(d), bool)
        for name, m in steps:
            keep &= m
            rows.append(dict(site=site, step=name, **fitstats(d[keep])))
        ideal.append(pd.DataFrame({"site": site, "hour": d.index[keep]}))
    if not rows:
        raise SystemExit("no flags files: run `cloudnet_quality.py flags` first")
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "steps.csv"), index=False, float_format="%.3g")
    pd.concat(ideal).to_csv(os.path.join(OUT, "ideal_hours.csv"), index=False)
    pd.set_option("display.width", 200)
    w = s.pivot_table(index="site", columns="step", values=["n", "r_log"], sort=False)
    order = list(dict.fromkeys(s["step"]))
    print("hours after each cumulative step:")
    print(w["n"][order].to_string(float_format="{:.0f}".format))
    print("\nr(log Nd) parcel vs observed after each step:")
    print(w["r_log"][order].to_string(float_format="{:.2f}".format))
    first = s.groupby("site").first()
    last = s.groupby("site").last()
    print("\nall hours -> fully screened (parcel / observed ratio, r, slope):")
    for site in first.index:
        f, l = first.loc[site], last.loc[site]
        print(f"  {site:16s} n {f['n']:5.0f} -> {l['n']:4.0f}   ratio {f['ratio']:.2f} -> {l['ratio']:.2f}   "
              f"r {f['r_log']:.2f} -> {l['r_log']:.2f}   slope {f['slope']:.2f} -> {l['slope']:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    xi = {k: i for i, k in enumerate(order)}
    cmap = plt.get_cmap("tab20")
    for k, (site, g) in enumerate(s.groupby("site")):
        g = g[g["n"] >= a.min_hours]
        x = [xi[v] for v in g["step"]]
        axs[0].plot(x, g["r_log"], "o-", color=cmap(k % 20), label=site, ms=4)
        axs[1].plot(x, g["ratio"], "o-", color=cmap(k % 20), ms=4)
    axs[0].set_ylabel("r(log Nd), parcel vs observed")
    axs[1].set(ylabel="median parcel / observed Nd", yscale="log")
    axs[1].axhline(1, color="k", lw=0.8)
    for ax in axs:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Cumulative screening of remote-sensing hours ({a.weighting} weighting; points with >= {a.min_hours} h)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "steps.png"), dpi=150)
    print(f"-> {os.path.relpath(OUT, HERE)}/steps.csv, steps.png, ideal_hours.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["flags", "screen"])
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--gap-gates", type=int, default=2, help="flags: liquid gaps up to this many gates are one layer")
    ap.add_argument("--weighting", choices=["pdf", "flux"], default="flux")
    ap.add_argument("--min-single", type=float, default=0.8, help="fraction of liquid profiles with one layer")
    ap.add_argument("--max-drizzle", type=float, default=0.1)
    ap.add_argument("--max-ice", type=float, default=0.1)
    ap.add_argument("--max-rain", type=float, default=0.0)
    ap.add_argument("--lwp-range", type=float, nargs=2, default=[30.0, 300.0], help="g m-2")
    ap.add_argument("--max-iqr", type=float, default=2.0, help="within-hour Q75/Q25 of retrieved Nd")
    ap.add_argument("--require-coupled", action="store_true")
    ap.add_argument("--min-hours", type=int, default=20, help="plot steps only with at least this many hours")
    a = ap.parse_args()
    {"flags": flags, "screen": screen}[a.step](a)


if __name__ == "__main__":
    main()
