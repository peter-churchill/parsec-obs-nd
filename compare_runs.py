#!/usr/bin/env python
"""Compare a sensitivity run directory with the default runs, hour by hour, on the hours both have.

For each site: the ratio test/base of
  Nd per run (criterion 3) and Smax, median over hours, at each fixed updraft
  updraft-weighted Nd (observed sigma_w, pdf and flux weighting, via weight_updrafts.py)
and the median number added above the measured spectrum (N_tail_cm3, N_coarse_cm3 from the test runs).

python compare_runs.py --test runs_tail_coarse --sites ena lasic mace-head pallas granada
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


def runs(rundir, site):
    f = sorted(glob.glob(os.path.join(rundir, site, "chunk_*.csv")))
    if not f:
        return None
    df = pd.concat([pd.read_csv(x, parse_dates=["hour"]) for x in f], ignore_index=True)
    df = df[df["status"] == "ok"].drop_duplicates(["hour", "w"], keep="last")
    if "Nd_c3" not in df:
        df["Nd_c3"] = df["AF_c3"] * df["N_input_cm3"]
    return df


def weighted(site, rundir, wt, out):
    wf = os.path.join(HERE, "updrafts", f"{site}_w_cloudbase.nc")
    cmd = [sys.executable, os.path.join(HERE, "weight_updrafts.py"), "--site", site, "--rundir", rundir,
           "--weighting", wt, "--w-file", wf, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   {site} {wt}: weighting failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:120]}")
        return None
    with xr.open_dataset(out) as d:
        return pd.Series(d["Nd_parcel"].values, index=pd.DatetimeIndex(d["hour"].values))


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if not len(x):
        return "   -"
    lx = np.log10(x)
    return f"{10 ** np.median(lx):.3f} ({10 ** np.percentile(lx, 10):.3f}-{10 ** np.percentile(lx, 90):.3f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="sensitivity run directory, e.g. runs_tail_coarse")
    ap.add_argument("--base", default=os.path.join(HERE, "runs"))
    ap.add_argument("--sites", nargs="*")
    a = ap.parse_args()
    test = os.path.join(HERE, a.test) if not os.path.isabs(a.test) else a.test
    tag = os.path.basename(test.rstrip("/"))
    out = os.path.join(HERE, "results", "sensitivity", tag)
    os.makedirs(out, exist_ok=True)
    sites = a.sites or sorted(d for d in os.listdir(test) if glob.glob(os.path.join(test, d, "chunk_*.csv")))
    rows = []
    for site in sites:
        T, B = runs(test, site), runs(a.base, site)
        if T is None or B is None:
            print(f"{site}: missing runs in {'test' if T is None else 'base'}")
            continue
        m = T.merge(B, on=["hour", "w"], suffixes=("_t", "_b"))
        hrs = m["hour"].nunique()
        print(f"\n{site}: {hrs} common hours; added above the spectrum: tail {T['N_tail_cm3'].median():.2f} cm-3, "
              f"coarse {T['N_coarse_cm3'].median():.2f} cm-3 (median; p90 {T['N_coarse_cm3'].quantile(.9):.2f}); "
              f"tail fits ok {np.mean(T.drop_duplicates('hour')['tail_flag'] == 'ok'):.0%}")
        print("     w    Nd test/base, median (10-90 %)       Smax test/base")
        for w, g in m.groupby("w"):
            print(f"   {w:4.2f}   {q(g['Nd_c3_t'] / g['Nd_c3_b']):28s}   {q(g['Smax_pct_t'] / g['Smax_pct_b']) if 'Smax_pct_t' in g else '-'}")
        row = dict(site=site, hours=hrs, N_tail=T["N_tail_cm3"].median(), N_coarse=T["N_coarse_cm3"].median())
        for wt in ("pdf", "flux"):
            nt = weighted(site, test, wt, os.path.join(out, f"{site}_test_{wt}.nc"))
            nb = weighted(site, a.base, wt, os.path.join(out, f"{site}_base_{wt}.nc"))
            if nt is not None and nb is not None:
                r = (nt / nb.reindex(nt.index)).dropna()
                print(f"   updraft-weighted ({wt}): Nd test/base {q(r)}")
                row[f"ratio_{wt}_median"] = 10 ** np.median(np.log10(r[r > 0])) if (r > 0).any() else np.nan
        rows.append(row)
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.4g")
        print(f"\n{t.to_string(index=False, float_format='{:.3f}'.format)}\n-> {os.path.relpath(out, HERE)}/summary.csv")


if __name__ == "__main__":
    main()
