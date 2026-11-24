#!/usr/bin/env python
"""Susceptibility profiles, closure diameters and activation diameters: parcel model against observations.

For each site with updraft-weighted parcel Nd (results/<site>_parcel_nd_obs_<weighting>.nc from compare_nd.py),
on the same hours and with the same N(>D) (the CCN(time, diameter) matrix of the observational built file):

  S(D)        susceptibility dln Nd / dln N(>D) at every cutoff D, by the observational recipe itself
              (arm_ena_nd_pnsd.susceptibility_table: log10, equal-number bins, regression through the bin
              medians), with 95 % day-block bootstrap intervals (bootstrap_by_day), for
              Nd = observed, parcel (pdf weighting) and parcel (flux weighting)
  D_Delta     closure diameter: where mean log Nd = mean log N(>D), with its bootstrap interval
  D_act       parcel activation diameter: 2 x PARSEC's smallest activated dry radius, averaged over each hour's
              updraft distribution with the same weighting as Nd; its median over the hours and the diameter
              N(>D) = the hours' mean log of N(>D_act) (comparable to D_Delta)
  D_act,obs   observed activation diameter where one exists (Zeppelin: cloud residuals vs whole air)

Writes results/susceptibility_parcel/<site>_profiles.csv (D, S and CI for each Nd), results/susceptibility_parcel/
summary.csv (one row per site: D_Delta, S(D_Delta), S(100 nm), D_act with intervals) and three figures:
  profiles.png      S(D) per site: observed (black, CI shaded) vs parcel pdf / flux, with D_Delta and D_act marked
  s_at_ddelta.png   S(D_Delta) vs D_Delta, observed and parcel, joined per site
  ddelta_vs_dact.png  D_Delta (observed, parcel) against the parcel activation diameter, with 1:1

The observational screens are those of the inputs (build_parcel_inputs.py); the profiles therefore cover the hours
with a spectrum and a parcel result, which can differ slightly from the published quicklook.

--by station pools the sites of each sites.STATIONS group (same aerosol, Nd observed two ways: ARM NDROP and
Cloudnet; in-cloud mountain station and Cloudnet below) into one profile, written to
results/susceptibility_parcel/stations/; each hour keeps its own member's parcel result, and the summary gives
the hours per Nd method (hours_ndrop, hours_cloudnet, hours_insitu).

--split coupling / temperature repeats everything per coupling class (coupled: |cloud base - surface LCL| <= 300 m,
decoupled, in-cloud station) or cloud-base temperature class (warm >= 0 C, supercooled -10..0 C, cold < -10 C),
into results/susceptibility_parcel/<split>_<group>/, with summary_<split>.csv and split_<split>.png comparing
the groups site by site. Sites or groups with fewer than --min-hours (30) matched hours are skipped.

Needs OBS_REPO (arm_ena_nd_pnsd.py and the built files) and scipy.

Examples
--------
python susceptibility_parcel.py --n-boot 200
python susceptibility_parcel.py --sites ena sgp hyytiala --n-boot 50
python susceptibility_parcel.py --split coupling --n-boot 200
python susceptibility_parcel.py --split temperature --n-boot 200
python susceptibility_parcel.py --by station --n-boot 200      # pool sites.STATIONS (e.g. SGP = NDROP + Cloudnet)
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
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
from sites import SITES, STATIONS, station_of  # noqa: E402

OUT = os.path.join(HERE, "results", "susceptibility_parcel")
HOURS = None            # {site: DatetimeIndex} from --hours-file
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_built(site, obs_repo):
    p = os.path.join(obs_repo, SITES[site]["built"])
    with xr.open_dataset(p) as ds:
        t = pd.DatetimeIndex(ds["time"].values).floor("h")
        ccn = ds["CCN"].values.astype(float)
        diam = ds["diameter"].values.astype(float)
    df = pd.DataFrame(ccn, index=t)
    return df[~df.index.duplicated()], diam


def parcel_dact(site, wt, rundir, hours):
    """Updraft-weighted parcel activation diameter (nm) per hour, weighted like Nd."""
    from weight_updrafts import gaussian_positive
    files = sorted(glob.glob(os.path.join(rundir, site, "chunk_*.csv")))
    wf = os.path.join(HERE, "updrafts", f"{site}_w_cloudbase.nc")
    if not files or not os.path.exists(wf):
        return pd.Series(np.nan, index=hours)
    df = pd.concat([pd.read_csv(f, parse_dates=["hour"]) for f in files], ignore_index=True)
    df = df[df["status"] == "ok"].drop_duplicates(["hour", "w"], keep="last")
    r = df.pivot_table(index="hour", columns="w", values="r_act_min_nm")
    wgrid = r.columns.to_numpy(float)
    with xr.open_dataset(wf) as d:
        wh = pd.DatetimeIndex(d["hour"].values)
        sig = pd.Series(d["sigma_w"].values, index=wh)
        mu = pd.Series(d["w_mean"].values if "w_mean" in d else 0.0, index=wh)
        W = pd.DataFrame(d["w"].values, index=wh) if "w" in d else None
    out = pd.Series(np.nan, index=hours)
    for h in hours:
        if h not in r.index:
            continue
        row = r.loc[h].to_numpy(float)
        ok = np.isfinite(row) & (row > 0)
        if ok.sum() < 3:
            continue
        w = W.loc[h].to_numpy(float) if (W is not None and h in W.index) else np.array([])
        w = w[np.isfinite(w) & (w > 0)]
        if len(w) >= 30:
            wq, pq = w, np.ones(len(w)) / len(w)
        else:
            s, m = sig.get(h, np.nan), mu.get(h, 0.0)
            if not (np.isfinite(s) and s > 0):
                continue
            wq, pq = gaussian_positive(s, 0.0 if not np.isfinite(m) else m)
            if wq is None:
                continue
        lw = np.log(np.clip(wq, wgrid[ok][0], wgrid[ok][-1]))
        dq = 2 * np.exp(np.interp(lw, np.log(wgrid[ok]), np.log(row[ok])))
        f = pq * (wq if wt == "flux" else 1)
        out[h] = np.sum(dq * f) / np.sum(f)
    return out


def interp_log(d, y, x):
    ok = np.isfinite(y)
    return float(np.interp(np.log10(x), np.log10(d[ok]), y[ok])) if ok.sum() > 1 and np.isfinite(x) else np.nan


def n_above_at(ccn, diam, dvals):
    """N(>D) per hour at an hour-specific D, interpolated in log D / log N."""
    out = np.full(len(dvals), np.nan)
    for i, dv in enumerate(dvals):
        c = ccn[i]
        ok = np.isfinite(c) & (c > 0)
        if ok.sum() > 1 and np.isfinite(dv):
            out[i] = 10 ** np.interp(np.log10(dv), np.log10(diam[ok]), np.log10(c[ok]))
    return out


def gather(site, a, group=None):
    """Matched hours of one site: N(>D) matrix, observed and parcel Nd, parcel activation diameters."""
    nd_files = {wt: os.path.join(HERE, "results", f"{site}_parcel_nd_obs_{wt}.nc") for wt in ("pdf", "flux")}
    if not all(os.path.exists(f) for f in nd_files.values()):
        print(f"{site}: no weighted parcel results (run compare_nd.py first)")
        return None
    ccn_df, diam = load_built(site, a.obs_repo)
    nd = {}
    for wt, f in nd_files.items():
        with xr.open_dataset(f) as d:
            s = pd.Series(d["Nd_parcel"].values, index=pd.DatetimeIndex(d["hour"].values))
            nd[f"parcel_{wt}"] = s
            nd["obs"] = pd.Series(d["Nd_obs"].values, index=s.index)
            thermo = pd.Series(d["thermo_source"].values.astype(str), index=s.index)
    hours = nd["obs"].index
    keep = (thermo != "placeholder") & nd["obs"].gt(0) & nd["parcel_pdf"].gt(0) & nd["parcel_flux"].gt(0)
    keep &= hours.isin(ccn_df.index)
    if HOURS is not None:
        keep &= hours.isin(HOURS.get(site, pd.DatetimeIndex([])))
    if group is not None:
        from compare_nd import classes
        c = classes(site, hours)
        keep &= (c[group[0]] == group[1]).values
    hours = hours[keep.values]
    g = dict(site=site, hours=hours, diam=diam, ccn=ccn_df.loc[hours].to_numpy(float),
             nd={k: v.loc[hours].to_numpy(float) for k, v in nd.items()},
             dact={wt: parcel_dact(site, wt, a.rundir, hours).to_numpy(float) for wt in ("pdf", "flux")},
             dact_obs=np.full(len(hours), np.nan))
    if site == "zeppelin" and os.path.exists(os.path.join(HERE, "results", "zeppelin_activation.nc")):
        with xr.open_dataset(os.path.join(HERE, "results", "zeppelin_activation.nc")) as d:
            g["dact_obs"] = pd.Series(d["d_act_int"].values,
                                      index=pd.DatetimeIndex(d["hour"].values)).reindex(hours).to_numpy(float)
    return g


def regrid(ccn, d_from, d_to):
    """N(>D) rows onto another diameter grid (log-log; NaN outside)."""
    if len(d_from) == len(d_to) and np.allclose(d_from, d_to):
        return ccn
    out = np.full((len(ccn), len(d_to)), np.nan)
    for i, c in enumerate(ccn):
        ok = np.isfinite(c) & (c > 0)
        if ok.sum() > 1:
            out[i] = 10 ** np.interp(np.log10(d_to), np.log10(d_from[ok]), np.log10(c[ok]), left=np.nan, right=np.nan)
    return out


def pool(name, parts):
    """One station from several sites' matched hours (rows kept per site, so an hour observed both ways counts
    twice, once per Nd method; the day-block bootstrap keeps them together)."""
    diam = parts[0]["diam"]
    return dict(site=name, members=[p["site"] for p in parts if len(p["hours"])],
                hours=pd.DatetimeIndex(np.concatenate([p["hours"].values for p in parts])),
                diam=diam, ccn=np.vstack([regrid(p["ccn"], p["diam"], diam) for p in parts]),
                nd={k: np.concatenate([p["nd"][k] for p in parts]) for k in parts[0]["nd"]},
                dact={wt: np.concatenate([p["dact"][wt] for p in parts]) for wt in ("pdf", "flux")},
                dact_obs=np.concatenate([p["dact_obs"] for p in parts]),
                method=np.concatenate([[SITES[p["site"]]["kind"]] * len(p["hours"]) for p in parts]))


def analyse(g, a, aq, group=None):
    site, hours, diam, ccn = g["site"], g["hours"], g["diam"], g["ccn"]
    members = g.get("members", [site])
    if len(hours) < a.min_hours:
        print(f"{site}: only {len(hours)} matched hours{'' if group is None else ' (' + group[1] + ')'}; skipped")
        return None
    order = np.argsort(hours.values, kind="stable")          # bootstrap_by_day expects time order
    hours, ccn = hours[order], ccn[order]
    nd = {k: v[order] for k, v in g["nd"].items()}
    times = hours.values
    prof = pd.DataFrame({"D_nm": diam})
    first = members[0]
    row = dict(site=site, group="all" if group is None else group[1], regime=SITES[first]["regime"],
               kind="+".join(sorted({SITES[m]["kind"] for m in members})), members="+".join(members),
               hours=len(hours), days=len(np.unique(hours.normalize())))
    if "method" in g:
        meth = g["method"][order]
        row.update({f"hours_{k}": int((meth == k).sum()) for k in np.unique(meth)})
    i100 = int(np.argmin(np.abs(diam - 100)))
    ntot = ccn[:, int(np.nanargmin(diam))]                   # N(>smallest diameter) = total number of the spectrum
    ntot = ntot[np.isfinite(ntot) & (ntot > 0)]
    if len(ntot):
        row.update(Ntot_median=np.median(ntot), Ntot_q25=np.percentile(ntot, 25), Ntot_q75=np.percentile(ntot, 75),
                   Ntot_Dmin_nm=float(np.nanmin(diam)),
                   N100_median=np.nanmedian(ccn[:, i100]))
    for key in ("obs", "parcel_pdf", "parcel_flux"):
        y = nd[key]
        tab, dd = aq.susceptibility_table(ccn, y, diam)
        sb, db = aq.bootstrap_by_day(times, ccn, y, diam, a.n_boot)
        ci = np.array([aq._ci(sb[:, i]) for i in range(len(diam))])
        prof[f"S_{key}"], prof[f"S_{key}_lo"], prof[f"S_{key}_hi"] = tab["S"].values, ci[:, 0], ci[:, 1]
        prof[f"delta_{key}"] = tab["delta_dex"].values
        dlo, dhi = aq._ci(db)
        row.update({f"Dd_{key}": dd, f"Dd_{key}_lo": dlo, f"Dd_{key}_hi": dhi,
                    f"S_Dd_{key}": interp_log(diam, tab["S"].values, dd),
                    f"S_Dd_{key}_lo": interp_log(diam, ci[:, 0], dd), f"S_Dd_{key}_hi": interp_log(diam, ci[:, 1], dd),
                    f"S100_{key}": tab["S"].values[i100], f"S100_{key}_lo": ci[i100, 0], f"S100_{key}_hi": ci[i100, 1],
                    f"Nd_median_{key}": np.median(y)})
    for wt in ("pdf", "flux"):
        v = g["dact"][wt][order]
        na = n_above_at(ccn, diam, v)
        ok = np.isfinite(na) & (na > 0)
        # diameter at which mean log N(>D) equals the hours' mean log N(>D_act): D_act on the D_Delta scale
        if ok.sum() > 10:
            target = np.log10(na[ok]).mean()
            m = np.log10(np.where(ccn[ok] > 0, ccn[ok], np.nan)).mean(axis=0)
            d_eq = 10 ** np.interp(-target, -m[np.isfinite(m)], np.log10(diam[np.isfinite(m)]))
        else:
            d_eq = np.nan
        row.update({f"Dact_{wt}_median": np.nanmedian(v), f"Dact_{wt}_q25": np.nanpercentile(v, 25),
                    f"Dact_{wt}_q75": np.nanpercentile(v, 75), f"Dact_{wt}_closure": d_eq})
    o = g["dact_obs"]
    if np.isfinite(o).any():
        row.update(Dact_obs_median=np.nanmedian(o), Dact_obs_q25=np.nanpercentile(o, 25),
                   Dact_obs_q75=np.nanpercentile(o, 75))
    os.makedirs(OUT, exist_ok=True)
    prof.to_csv(os.path.join(OUT, f"{site}_profiles.csv"), index=False, float_format="%.4g")
    print(f"   [{row['group']}]", end=" ")
    print(f"{site:16s} {len(hours):5d} h  D_Delta obs {row['Dd_obs']:5.0f}  parcel pdf {row['Dd_parcel_pdf']:5.0f} "
          f"flux {row['Dd_parcel_flux']:5.0f} nm | S(D_Delta) obs {row['S_Dd_obs']:.2f} pdf {row['S_Dd_parcel_pdf']:.2f} "
          f"flux {row['S_Dd_parcel_flux']:.2f} | D_act pdf {row['Dact_pdf_median']:.0f} (closure-equiv. "
          f"{row['Dact_pdf_closure']:.0f}) nm" + (f"  [{row['members']}]" if len(members) > 1 else ""), flush=True)
    return row, prof


def unit_analysis(unit, a, aq, group=None):
    """unit: a site, or with --by station a station name whose member sites are pooled."""
    members = STATIONS.get(unit, [unit]) if a.by == "station" else [unit]
    parts = [p for p in (gather(m, a, group) for m in members) if p is not None]
    if not parts:
        return None
    g = parts[0] if (len(parts) == 1 and a.by != "station") else pool(unit, parts)
    return analyse(g, a, aq, group)


def plots(rows, profs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    col = {"obs": "k", "parcel_pdf": "#1f77b4", "parcel_flux": "#ff7f0e"}
    lab = {"obs": "observed", "parcel_pdf": "parcel (pdf)", "parcel_flux": "parcel (flux)"}

    def logx(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 300]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    n = len(rows)
    nc = 5
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.7 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, r, p in zip(axs.flat, rows, profs):
        d = p["D_nm"].to_numpy(float)
        ax.fill_between(d, p["S_obs_lo"], p["S_obs_hi"], color="0.85", lw=0)
        for k in ("obs", "parcel_pdf", "parcel_flux"):
            ax.plot(d, p[f"S_{k}"], color=col[k], lw=1.4 if k == "obs" else 1.1, label=lab[k])
            if np.isfinite(r[f"Dd_{k}"]):
                ax.axvline(r[f"Dd_{k}"], color=col[k], ls="--", lw=0.8)
        if np.isfinite(r["Dact_pdf_closure"]):
            ax.axvline(r["Dact_pdf_closure"], color=col["parcel_pdf"], ls=":", lw=1.2)
        if np.isfinite(r.get("Dact_obs_median", np.nan)):
            ax.axvline(r["Dact_obs_median"], color="k", ls=":", lw=1.2)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axhline(1, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{r['site']} ({r['hours']} h)", fontsize=9)
        logx(ax)
        ax.set_ylim(-0.5, 1.5)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("cutoff diameter (nm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("S(D)")
    axs.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Susceptibility profiles: observed (shaded: 95 % day-block CI) vs parcel. "
                 "Dashed: D$_\\Delta$; dotted: activation diameter (blue parcel, black observed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profiles.png"), dpi=140)

    comparison_figures(pd.DataFrame(rows), OUT)
    print(f"-> {os.path.relpath(OUT, HERE)}/profiles.png, s_at_ddelta.png, ddelta_vs_dact.png, ddelta_vs_ntot.png")


REGIME_COLOURS = {"Marine": "#1f77b4", "Boreal": "#2ca02c", "Arctic": "#9467bd", "Continental": "#d62728"}


def _err(t, c, lo, hi):
    """asymmetric error bars (2, n) from CI columns, NaN-safe and non-negative"""
    if lo not in t or hi not in t:
        return None
    e = np.vstack([t[c] - t[lo], t[hi] - t[c]]).astype(float)
    return np.where(np.isfinite(e), np.clip(e, 0, None), 0)


def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return ""
    r, p = spearmanr(x[ok], y[ok])
    return f"Spearman ρ = {r:.2f} (p = {p:.2g}, n = {ok.sum()})"


def comparison_figures(t, out):
    """Observed (left) and parcel (right) side by side, same axes, one point per site coloured by regime:
    s_at_ddelta.png     S(D_Delta) vs D_Delta, with day-block bootstrap 95 % intervals on both
    ddelta_vs_dact.png  D_Delta vs the parcel activation diameter (closure-equivalent), 1:1 line;
                        observed activation diameter as a star where one exists (Zeppelin)
    ddelta_vs_ntot.png  D_Delta vs aerosol loading: median total number N(>D_min) of the matched hours (IQR bars)
    Parcel panels show pdf weighting (filled) and flux weighting (open)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    def logax(ax, which="x", ticks=(20, 50, 100, 200, 300)):
        getattr(ax, f"set_{which}scale")("log")
        a = ax.xaxis if which == "x" else ax.yaxis
        a.set_major_locator(FixedLocator(list(ticks)))
        a.set_major_formatter(ScalarFormatter())
        a.set_minor_formatter(NullFormatter())

    t = t.reset_index(drop=True)
    col = [REGIME_COLOURS.get(r, "0.5") for r in t["regime"]]
    reg_handles = [Line2D([], [], marker="o", ls="", color=c, mec="k", label=r) for r, c in REGIME_COLOURS.items()
                   if r in set(t["regime"])]
    wt_handles = [Line2D([], [], marker="o", ls="", color="0.4", mec="k", label="pdf weighting"),
                  Line2D([], [], marker="s", ls="", mfc="white", mec="0.4", mew=1.5, label="flux weighting")]
    ekw = dict(ls="", elinewidth=0.7, capsize=0, alpha=0.6, zorder=1)

    def points(ax, x, y, xe=None, ye=None, marker="o", filled=True, label=True):
        for i in range(len(t)):
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                continue
            ax.errorbar(x[i], y[i], xerr=None if xe is None else xe[:, [i]], yerr=None if ye is None else ye[:, [i]],
                        color=col[i], **ekw)
            ax.plot(x[i], y[i], marker, color=col[i] if filled else "white", mec="k" if filled else col[i],
                    mew=0.8 if filled else 1.5, ms=7, zorder=3)
            if label:
                ax.annotate(t["site"][i], (x[i], y[i]), xytext=(4, 3), textcoords="offset points", fontsize=6.5)

    def finish(fig, axs, name, title):
        axs[0].legend(handles=reg_handles, fontsize=7, loc="best")
        axs[1].legend(handles=reg_handles + wt_handles, fontsize=7, loc="best")
        for ax in axs:
            ax.grid(alpha=0.3)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, name), dpi=150)
        plt.close(fig)

    g = lambda c: t[c].to_numpy(float) if c in t else np.full(len(t), np.nan)

    # --- S(D_Delta) vs D_Delta
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    points(axs[0], g("Dd_obs"), g("S_Dd_obs"), _err(t, "Dd_obs", "Dd_obs_lo", "Dd_obs_hi"),
           _err(t, "S_Dd_obs", "S_Dd_obs_lo", "S_Dd_obs_hi"))
    for k, m, f in (("parcel_pdf", "o", True), ("parcel_flux", "s", False)):
        points(axs[1], g(f"Dd_{k}"), g(f"S_Dd_{k}"), _err(t, f"Dd_{k}", f"Dd_{k}_lo", f"Dd_{k}_hi"),
               _err(t, f"S_Dd_{k}", f"S_Dd_{k}_lo", f"S_Dd_{k}_hi"), m, f, label=f)
    for ax, ttl in zip(axs, ("observed", "parcel model")):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axhline(1, color="0.6", lw=0.8, ls=":")
        logax(ax)
        ax.set(xlabel="closure diameter D$_\\Delta$ (nm)", title=ttl)
    axs[0].set_ylabel("S(D$_\\Delta$)")
    finish(fig, axs, "s_at_ddelta.png", "Susceptibility at the closure diameter (bars: 95 % day-block bootstrap)")

    # --- D_Delta vs activation diameter
    lim = [15, 400]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    points(axs[0], g("Dact_pdf_closure"), g("Dd_obs"), None, _err(t, "Dd_obs", "Dd_obs_lo", "Dd_obs_hi"))
    zx, zy = g("Dact_obs_median"), g("Dd_obs")
    for i in range(len(t)):
        if np.isfinite(zx[i]) and np.isfinite(zy[i]):
            axs[0].plot(zx[i], zy[i], "*", color=col[i], mec="k", ms=14, zorder=4)
            axs[0].annotate("observed D$_{act}$", (zx[i], zy[i]), xytext=(4, -10), textcoords="offset points", fontsize=6.5)
    for k, wt, m, f in (("parcel_pdf", "pdf", "o", True), ("parcel_flux", "flux", "s", False)):
        points(axs[1], g(f"Dact_{wt}_closure"), g(f"Dd_{k}"), None, _err(t, f"Dd_{k}", f"Dd_{k}_lo", f"Dd_{k}_hi"),
               m, f, label=f)
    axs[0].set_title("observed D$_\\Delta$\n" + _spearman(g("Dact_pdf_closure"), g("Dd_obs")), fontsize=9)
    axs[1].set_title("parcel D$_\\Delta$\n" + _spearman(g("Dact_pdf_closure"), g("Dd_parcel_pdf")) + " (pdf)", fontsize=9)
    for ax in axs:
        ax.plot(lim, lim, "k-", lw=0.8)
        logax(ax, "x")
        logax(ax, "y")
        ax.set(xlim=lim, ylim=lim, xlabel="parcel activation diameter, closure-equivalent (nm)")
    axs[0].set_ylabel("closure diameter D$_\\Delta$ (nm)")
    finish(fig, axs, "ddelta_vs_dact.png", "Closure diameter vs activation diameter (1:1 line; star: observed activation diameter)")

    # --- D_Delta vs aerosol loading
    x = g("Ntot_median")
    xe = _err(t, "Ntot_median", "Ntot_q25", "Ntot_q75")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True)
    points(axs[0], x, g("Dd_obs"), xe, _err(t, "Dd_obs", "Dd_obs_lo", "Dd_obs_hi"))
    for k, m, f in (("parcel_pdf", "o", True), ("parcel_flux", "s", False)):
        points(axs[1], x, g(f"Dd_{k}"), xe, _err(t, f"Dd_{k}", f"Dd_{k}_lo", f"Dd_{k}_hi"), m, f, label=f)
    axs[0].set_title("observed\n" + _spearman(x, g("Dd_obs")), fontsize=9)
    axs[1].set_title("parcel model\n" + _spearman(x, g("Dd_parcel_pdf")) + " (pdf)", fontsize=9)
    dmin = np.nanmedian(g("Ntot_Dmin_nm"))
    for ax in axs:
        ax.set_xscale("log")
        logax(ax, "y")
        ax.set(ylim=lim, xlabel=f"aerosol loading N$_{{tot}}$ (N > {dmin:.0f} nm, cm$^{{-3}}$; median, IQR bars)")
    axs[0].set_ylabel("closure diameter D$_\\Delta$ (nm)")
    finish(fig, axs, "ddelta_vs_ntot.png", "Closure diameter vs aerosol loading")


SPLITS = {"coupling": ("coupling", ["coupled", "decoupled", "in-cloud station"]),
          "temperature": ("temp", ["warm (>= 0 C)", "supercooled (-10..0 C)", "cold (< -10 C)"])}


def split_figure(t, split, path):
    """S(D_Delta) observed (filled) and parcel flux (open) per site and group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = SPLITS[split][1]
    col = dict(zip(groups, ["#1f77b4", "#d62728", "#2ca02c"]))
    sites = sorted(t["site"].unique())
    x = {s: i for i, s in enumerate(sites)}
    fig, axs = plt.subplots(2, 1, figsize=(max(8, 0.55 * len(sites)), 7.5), sharex=True)
    for g in groups:
        u = t[t["group"] == g]
        off = (groups.index(g) - 1) * 0.22
        xs = np.array([x[s] for s in u["site"]]) + off
        axs[0].plot(xs, u["S_Dd_obs"], "o", color=col[g], mec="k", ms=7, label=f"{g}: observed")
        axs[0].plot(xs, u["S_Dd_parcel_flux"], "o", mfc="white", mec=col[g], mew=1.5, ms=7, label=f"{g}: parcel (flux)")
        axs[1].plot(xs, u["Dd_obs"] / u["Dact_flux_closure"], "s", color=col[g], mec="k", ms=6, label=g)
    axs[0].axhline(0, color="0.6", lw=0.6)
    axs[0].axhline(1, color="0.6", lw=0.6, ls=":")
    axs[0].set_ylabel("S(D$_\\Delta$)")
    axs[0].legend(fontsize=7, ncol=3)
    axs[1].axhline(1, color="k", lw=0.8)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("observed D$_\\Delta$ / parcel activation diameter")
    axs[1].legend(fontsize=7, ncol=3)
    axs[1].set_xticks(range(len(sites)))
    axs[1].set_xticklabels(sites, rotation=60, ha="right", fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Split by {split}: susceptibility at the closure diameter and closure vs activation diameter")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--obs-repo", default=os.environ.get("OBS_REPO"))
    ap.add_argument("--rundir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--split", choices=["none", "coupling", "temperature"], default="none",
                    help="also compute everything per coupling class (|cloud base - LCL| <= 300 m) or per cloud-base "
                         "temperature class, into results/susceptibility_parcel/<split>_<group>/")
    ap.add_argument("--min-hours", type=int, default=30, help="minimum matched hours per site (and group)")
    ap.add_argument("--hours-file", help="CSV with columns site, hour: use only these hours (e.g. "
                                          "results/cloudnet_quality/ideal_hours.csv); sites not in it are skipped")
    ap.add_argument("--tag", help="write to results/susceptibility_parcel/<tag>/ (default: the hours file's name)")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw s_at_ddelta / ddelta_vs_dact / ddelta_vs_ntot from the existing summary.csv")
    ap.add_argument("--by", choices=["site", "station"], default="site",
                    help="station: pool the sites of each sites.STATIONS group (same aerosol, different Nd "
                         "method) into one profile; output in results/susceptibility_parcel/stations/")
    a = ap.parse_args()
    global HOURS
    if a.hours_file:
        h = pd.read_csv(a.hours_file, parse_dates=["hour"])
        HOURS = {k: pd.DatetimeIndex(g["hour"]) for k, g in h.groupby("site")}
        OUT = os.path.join(OUT, a.tag or os.path.splitext(os.path.basename(a.hours_file))[0])
        if not a.sites:
            a.sites = sorted(HOURS)
    elif a.tag:
        OUT = os.path.join(OUT, a.tag)
    if a.plots_only:
        out = os.path.join(OUT, "stations") if a.by == "station" else OUT
        t = pd.read_csv(os.path.join(out, "summary.csv"))
        if "Ntot_median" not in t:
            print("summary.csv has no Ntot_median yet: rerun once without --plots-only for ddelta_vs_ntot.png")
        comparison_figures(t, out)
        print(f"-> {os.path.relpath(out, HERE)}/s_at_ddelta.png, ddelta_vs_dact.png, ddelta_vs_ntot.png")
        return
    if not a.obs_repo:
        ap.error("give --obs-repo or set OBS_REPO")
    sys.path.insert(0, a.obs_repo)
    import arm_ena_nd_pnsd as aq
    sites = a.sites or sorted({os.path.basename(f).split("_parcel_nd_obs_")[0]
                               for f in glob.glob(os.path.join(HERE, "results", "*_parcel_nd_obs_pdf.nc"))})
    if a.by == "station":
        sites = sorted({s if s in STATIONS else station_of(s) for s in sites})   # --sites takes sites or stations
        OUT = os.path.join(OUT, "stations")
    base = OUT
    runs = [(None, base)] if a.split == "none" else \
        [((SPLITS[a.split][0], g), os.path.join(base, f"{a.split}_{g.split(' ')[0]}")) for g in SPLITS[a.split][1]]
    allrows = []
    for group, out in runs:
        OUT = out
        if group is not None:
            print(f"\n=== {a.split}: {group[1]} ===")
        rows, profs = [], []
        for s in sites:
            r = unit_analysis(s, a, aq, group)
            if r:
                rows.append(r[0])
                profs.append(r[1])
        if not rows:
            print("   nothing for this group")
            continue
        os.makedirs(OUT, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.4g")
        print(f"-> {os.path.relpath(OUT, HERE)}/summary.csv")
        plots(rows, profs)
        allrows += rows
    if a.split != "none" and allrows:
        t = pd.DataFrame(allrows)
        t.to_csv(os.path.join(base, f"summary_{a.split}.csv"), index=False, float_format="%.4g")
        split_figure(t, a.split, os.path.join(base, f"split_{a.split}.png"))
        print(f"-> {os.path.relpath(base, HERE)}/summary_{a.split}.csv, split_{a.split}.png")


if __name__ == "__main__":
    main()
