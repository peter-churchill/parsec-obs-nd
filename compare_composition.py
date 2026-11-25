#!/usr/bin/env python
"""Observed vs CAMS aerosol composition on the same cloud hours.

Compares inputs/<site>_parcel_inputs.nc (observation-based composition from build_composition.py)
with inputs/<site>_parcel_inputs_cams.nc (the same hours, CAMS only: build_composition.py --cams-only
--suffix _cams), on the hours whose observed tier is "acsm" or "filter". By default only hours where
the CAMS file has 3-hourly EAC4 (cams3h) are used; --monthly also admits EAC4 monthly means (camsmon).

Quantities (ug m-3 unless noted), and the hours on which each is an independent measurement:
  inorganic   sulfate + ammonium (+ nitrate, chloride from the ACSM) as the three PARSEC sulfate
              columns. CAMS has no nitrate, so a high obs/CAMS ratio at nitrate-rich sites is expected.
              acsm, filter
  organics    organic matter                                        acsm only (filter organics are
              CAMS OM/SO4 x measured sulfate, so not independent)
  org_frac    organics / (inorganic + organics + BC), 1             acsm only
  bc          black carbon                                          hours with measured eBC
  seasalt     submicron dry sea salt                                filter only (from measured Na)
  kappa       bulk kappa of the dry mass fractions, 1               acsm, filter

Output: results/composition_obs_vs_cams.csv (per site, tier and quantity: hours, medians, median
CAMS/obs ratio with quartiles, r of log values) and results/composition_obs_vs_cams.png.

Examples
--------
python build_composition.py --cams-only --suffix _cams     # once, after new CAMS files arrive
python compare_composition.py
python compare_composition.py --sites pallas --monthly
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
INORG = ["m_h2so4", "m_nh4hso4", "m_nh42so4"]
QUANT = {  # name: (obs tiers where independent, needs measured BC)
    "inorganic": (("acsm", "filter"), False),
    "organics": (("acsm",), False),
    "org_frac": (("acsm",), False),
    "bc": (("acsm", "filter"), True),
    "seasalt": (("filter",), False),
    "kappa": (("acsm", "filter"), False),
}


def table(ds):
    src = pd.Series(ds["comp_source"].values.astype(str))
    d = pd.DataFrame({"hour": pd.DatetimeIndex(ds["hour"].values),
                      "tier": src.str.split("+").str[0].values,
                      "bc_obs": src.str.contains("bc_obs").values,
                      "cams_tag": src.str.split("+").str[-1].values})
    if not all(k in ds for k in INORG + ["m_org", "m_bc", "m_seasalt", "kappa_bulk"]):
        return None
    d["inorganic"] = sum(ds[k].values for k in INORG)
    d["organics"] = ds["m_org"].values
    d["bc"] = ds["m_bc"].values
    d["seasalt"] = ds["m_seasalt"].values
    d["org_frac"] = d["organics"] / (d["inorganic"] + d["organics"] + d["bc"])
    d["kappa"] = ds["kappa_bulk"].values
    return d


def pair(site, monthly):
    p_obs = os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")
    p_cams = os.path.join(HERE, "inputs", f"{site}_parcel_inputs_cams.nc")
    if not (os.path.exists(p_obs) and os.path.exists(p_cams)):
        return None, "no _cams file (run build_composition.py --cams-only --suffix _cams)"
    with xr.open_dataset(p_obs) as a, xr.open_dataset(p_cams) as b:
        o, c = table(a), table(b)
    if o is None or c is None:
        return None, "inputs file has no composition masses (run build_composition.py first)"
    m = o.merge(c, on="hour", suffixes=("_obs", "_cams"))
    ok_tags = ("cams3h", "camsmon") if monthly else ("cams3h",)
    m = m[m["tier_obs"].isin(["acsm", "filter"]) & m["cams_tag_cams"].isin(ok_tags)]
    if m.empty:
        return None, (f"no hours with observed acsm/filter composition and CAMS "
                      f"{'3-hourly or monthly' if monthly else '3-hourly'} data")
    return m, ""


def stats(m, site):
    rows = []
    for q, (tiers, need_bc) in QUANT.items():
        for tier in ("acsm", "filter"):
            if tier not in tiers:
                continue
            g = m[(m["tier_obs"] == tier) & (m["bc_obs_obs"] if need_bc else True)]
            x, y = g[f"{q}_obs"].to_numpy(float), g[f"{q}_cams"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if ok.sum() < 5:
                continue
            lx, ly = np.log10(x[ok]), np.log10(y[ok])
            lr = ly - lx
            rows.append(dict(site=site, tier=tier, quantity=q, hours=int(ok.sum()),
                             obs_median=np.median(x[ok]), cams_median=np.median(y[ok]),
                             ratio_cams_obs=10**np.median(lr), ratio_q25=10**np.percentile(lr, 25),
                             ratio_q75=10**np.percentile(lr, 75),
                             r_log=np.corrcoef(lx, ly)[0, 1] if ok.sum() > 2 else np.nan,
                             years=f"{g['hour'].min().year}-{g['hour'].max().year}"))
    return rows


def plot(pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qs = ["inorganic", "organics", "org_frac", "kappa"]
    sites = list(pairs)
    fig, axs = plt.subplots(len(sites), len(qs), figsize=(3.3 * len(qs), 3.1 * len(sites)), squeeze=False)
    for i, s in enumerate(sites):
        m = pairs[s]
        for j, q in enumerate(qs):
            ax = axs[i, j]
            tiers = QUANT[q][0]
            for tier, col in (("acsm", "#1f77b4"), ("filter", "#d62728")):
                if tier not in tiers:
                    continue
                g = m[m["tier_obs"] == tier]
                if len(g):
                    ax.scatter(g[f"{q}_obs"], g[f"{q}_cams"], s=4, alpha=0.4, color=col, label=tier, rasterized=True)
            v = pd.concat([m[f"{q}_obs"], m[f"{q}_cams"]])
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) == 0:
                ax.set_visible(False)
                continue
            lim = [v.min() * 0.8, v.max() * 1.25]
            ax.plot(lim, lim, "k-", lw=0.7)
            if q not in ("kappa", "org_frac"):
                ax.set(xscale="log", yscale="log")
                for f in (0.5, 2):
                    ax.plot(lim, [l * f for l in lim], "k:", lw=0.5)
            ax.set(xlim=lim, ylim=lim, title=f"{s}: {q}", xlabel="observed", ylabel="CAMS")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, markerscale=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--monthly", action="store_true", help="also use hours with only EAC4 monthly means")
    a = ap.parse_args()
    sites = a.sites or sorted(os.path.basename(f)[:-len("_parcel_inputs.nc")]
                              for f in glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))
    rows, pairs = [], {}
    for s in sites:
        m, why = pair(s, a.monthly)
        if m is None:
            print(f"{s}: {why}")
            continue
        print(f"{s}: {len(m)} hours ({m['tier_obs'].value_counts().to_dict()}, "
              f"CAMS {m['cams_tag_cams'].value_counts().to_dict()})")
        pairs[s] = m
        rows += stats(m, s)
    if not rows:
        raise SystemExit("nothing to compare")
    t = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    t.to_csv(os.path.join(RES, "composition_obs_vs_cams.csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print("\nCAMS vs observed composition (ratio = CAMS / observed, median of hourly ratios):")
    print(t.to_string(index=False, float_format="{:.2f}".format))
    out = os.path.join(RES, "composition_obs_vs_cams.png")
    plot(pairs, out)
    print(f"-> results/composition_obs_vs_cams.csv, {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
