#!/usr/bin/env python
"""Observed Nd against observed Nd on shared hours: how far two retrievals of the same cloud disagree, without
the parcel model; and, where the parcel model ran for those hours, which of the two it follows.

Pairs (matched-hour CSVs written in the observation repo, $OBS_REPO):
  zeppelin     results/ny-alesund/nyalesund_cloudnet_vs_zeppelin_insitu.csv   Nd_is (in situ) vs Nd (Cloudnet)
  sgp          results/sgp/sgp_cloudnet_vs_ndrop.csv                           Nd_ndrop vs Nd_cloudnet
  ena          results/ena/ena_cloudnet_vs_ndrop.csv  (if made with nd_compare.py)
(Pallas / Kenttarova cannot be paired: Pallas in situ ends 2015, the Kenttarova Cloudnet file covers 2023.)

For each pair, on hours with both Nd > 0 and cloud_frac >= --min-cloud-frac (Cloudnet side):
  n, Pearson r of log Nd, median ratio B/A with IQR, SD of log10(B/A) (dex): the combined random error of the
  two retrievals if neither is biased in a way that varies with Nd; plus a subset (Zeppelin: Cloudnet liquid
  base below the station, 474 m, i.e. the station inside the cloud Cloudnet sees).
Where parcel Nd exists for the same hours (results/nd_comparison_hours.csv, observed-updraft variant):
  r of log parcel Nd with each retrieval, and a triple-collocation estimate of each one's random error
  (log space; assumes independent errors, which is optimistic: in situ Nd and the parcel input share the total
  inlet, NDROP and Cloudnet share the MWR LWP).
Writes results/obs_vs_obs/summary.csv and obs_vs_obs.png.

python obs_vs_obs.py
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "obs_vs_obs")
# name: (csv in OBS_REPO, column A, label A, column B, label B, parcel site for A, parcel site for B, subset)
PAIRS = {
    "zeppelin": ("results/ny-alesund/nyalesund_cloudnet_vs_zeppelin_insitu.csv", "Nd_is", "Zeppelin in situ",
                 "Nd", "Ny-Alesund Cloudnet", "zeppelin", "ny-alesund", ("cloud_base_height", 474.0)),
    "sgp": ("results/sgp/sgp_cloudnet_vs_ndrop.csv", "Nd_ndrop", "SGP NDROP", "Nd_cloudnet", "SGP Cloudnet",
            "sgp", "sgp_cloudnet", None),
    "ena": ("results/ena/ena_cloudnet_vs_ndrop.csv", "Nd_ndrop", "ENA NDROP", "Nd_cloudnet", "ENA Cloudnet",
            "ena", "ena_cloudnet", None),
}


def stats(a, b):
    la, lb = np.log10(a), np.log10(b)
    d = lb - la
    return dict(n=len(a), r_log=np.corrcoef(la, lb)[0, 1] if len(a) > 2 else np.nan,
                ratio_median=10 ** np.median(d), ratio_q25=10 ** np.percentile(d, 25),
                ratio_q75=10 ** np.percentile(d, 75), sd_log_ratio=np.std(d, ddof=1) if len(a) > 2 else np.nan)


def triple(x, y, z):
    """Random-error SD (dex) of x, y, z by triple collocation (covariance form)."""
    c = np.cov(np.vstack([x, y, z]))
    e = [c[0, 0] - c[0, 1] * c[0, 2] / c[1, 2], c[1, 1] - c[0, 1] * c[1, 2] / c[0, 2],
         c[2, 2] - c[0, 2] * c[1, 2] / c[0, 1]]
    return [np.sqrt(v) if v > 0 else 0.0 for v in e]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--min-cloud-frac", type=float, default=0.0,
                    help="Cloudnet cloud_frac screen on the paired hours (the paired CSVs are unscreened)")
    ap.add_argument("--weighting", choices=["pdf", "flux"], default="flux")
    a = ap.parse_args()
    if not a.obs_repo:
        ap.error("give --obs-repo or set OBS_REPO")
    os.makedirs(OUT, exist_ok=True)
    hp = os.path.join(HERE, "results", "nd_comparison_hours.csv")
    par = None
    if os.path.exists(hp):
        par = pd.read_csv(hp, parse_dates=["hour"])
        par = par[(par["variant"] == "obs") & (par["weighting"] == a.weighting)]
    rows, panels = [], []
    for name, (rel, ca, la, cb, lb, sa, sb, sub) in PAIRS.items():
        p = os.path.join(a.obs_repo, rel)
        if not os.path.exists(p):
            print(f"{name}: no {rel}")
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).floor("h")
        df = df[(df[ca] > 0) & (df[cb] > 0)]
        cf = next((c for c in ("cloud_frac", "cloud_frac_cloudnet") if c in df), None)
        if cf and a.min_cloud_frac > 0:
            df = df[df[cf] >= a.min_cloud_frac]
        sets = [("all", df)]
        if sub and sub[0] in df:
            sets.append((f"{sub[0]} < {sub[1]:.0f} m", df[df[sub[0]] < sub[1]]))
        for lab, d in sets:
            if len(d) < 3:
                continue
            r = dict(pair=name, subset=lab, A=la, B=lb, **stats(d[ca].values, d[cb].values))
            if par is not None:
                pa = par[par["site"] == sa].set_index("hour")["Nd_parcel"]
                pb = par[par["site"] == sb].set_index("hour")["Nd_parcel"]
                pn = pa.combine_first(pb)                  # same aerosol and nearly the same parcel Nd either way
                j = d.join(pn.rename("parcel"), how="inner").dropna(subset=["parcel"])
                j = j[j["parcel"] > 0]
                if len(j) >= 10:
                    x, y, z = np.log10(j[ca]), np.log10(j[cb]), np.log10(j["parcel"])
                    ea, eb, ep = triple(x.values, y.values, z.values)
                    r.update(n_parcel=len(j), r_parcel_A=np.corrcoef(z, x)[0, 1], r_parcel_B=np.corrcoef(z, y)[0, 1],
                             err_A_dex=ea, err_B_dex=eb, err_parcel_dex=ep)
            rows.append(r)
            panels.append((name, lab, la, lb, d[ca].values, d[cb].values))
    if not rows:
        raise SystemExit("no pairs found")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.3g")
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format="{:.2f}".format))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.4), squeeze=False)
    for ax, (name, lab, la, lb, x, y) in zip(axs[0], panels):
        lo = 10 ** np.floor(np.log10(min(x.min(), y.min())))
        hi = 10 ** np.ceil(np.log10(max(x.max(), y.max())))
        ax.scatter(x, y, s=10, alpha=0.6, color="#1f77b4", lw=0)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        for f in (0.5, 2):
            ax.plot([lo, hi], [lo * f, hi * f], "k:", lw=0.6)
        s = stats(x, y)
        ax.text(0.03, 0.97, f"n = {s['n']}\nB/A {s['ratio_median']:.2f} ({s['ratio_q25']:.2f}-{s['ratio_q75']:.2f})\n"
                f"r(log) = {s['r_log']:.2f}\nSD log(B/A) = {s['sd_log_ratio']:.2f} dex", transform=ax.transAxes,
                va="top", fontsize=7.5, bbox=dict(boxstyle="round", fc="white", ec="0.8"))
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=f"A: {la} Nd (cm$^{{-3}}$)",
               ylabel=f"B: {lb} Nd (cm$^{{-3}}$)", title=f"{name} ({lab})")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "obs_vs_obs.png"), dpi=150)
    print(f"-> {os.path.relpath(OUT, HERE)}/summary.csv, obs_vs_obs.png")


if __name__ == "__main__":
    main()
