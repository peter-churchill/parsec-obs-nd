#!/usr/bin/env python
"""How does droplet number change with height in the parcel runs, and does the sampling height matter?

The production runs keep only end-of-run numbers (150 m above cloud base). This reruns a handful of hours per
site at a few updrafts, deeper (--cloud-depth, default 600 m), and keeps PARSEC's NetCDF output to read the
profiles through the ascent:
  height above the start, supersaturation, and the activated number for PARSEC's four criteria
  (fa_act, across all aerosol: N1 wet r > 1 um, N2 Sc < Smax, N3 r > rcrit, N4 > smallest activated),
  per volume (cm-3) and per mass of air (number / air density, normalised at 150 m), since an adiabatic parcel
  conserves droplets per kg while its density falls with height.
Reports per run: height of peak supersaturation, and N at 25/50/100/150/300/600 m relative to 150 m (the height
the comparison uses). Writes results/parcel_profiles/profiles.csv, summary.csv and profiles.png.

python parcel_profiles.py inspect --site puijo --parsec <PARSEC dir>        # one run: variables and shapes
python parcel_profiles.py run --sites puijo pallas lasic ena --parsec <PARSEC dir> --hours 3

Run it on a compute node (interactive -A naiss2026-1-3-cpu -p cpu -t 01:00:00 -n 1 -c 1), with
PARSEC_LD_LIBRARY_PATH set as in parsec_array.sbatch.
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parsec_run as P  # noqa: E402

OUT = os.path.join(HERE, "results", "parcel_profiles")
LEVELS = [25, 50, 100, 150, 300, 600]


def one_run(site, ih, w, a, keep_nc=None):
    ds = xr.open_dataset(os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc"))
    spec = ds["dNdlogDp"].values[ih]
    bins = P.spectrum_to_bins(spec, ds["diameter"].values, 2)
    T0, P0 = float(ds["T0"].values[ih]), float(ds["P0"].values[ih])
    mf = ds["massfrac"].values[ih]
    hour = pd.Timestamp(ds["hour"].values[ih])
    ds.close()
    wd = tempfile.mkdtemp(prefix=f"prof_{site}_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        tmpl = os.path.join(a.parsec, "inputs")
        tlim = P.write_inputs(wd, tmpl, bins, mf, T0, P0, w, a.rh0, a.cloud_depth, 1.0, 0.1)
        rc, err, wall = P.run_one(os.path.join(a.parsec, "parsec-ufo"), wd, timeout=max(1200, 20 * tlim))
        ncs = sorted(glob.glob(os.path.join(wd, "outputs", "**", "*.nc"), recursive=True))
        if rc != 0 or not ncs:
            print(f"   {site} {hour} w {w}: run failed (rc {rc}) {err[-120:]}")
            return None, hour
        if keep_nc:
            for f in ncs:
                shutil.copy(f, keep_nc)
        return ncs, hour
    finally:
        if not keep_nc:
            pass
        globals().setdefault("_wds", []).append(wd)


def read_profile(ncs):
    """height (m), supersaturation (%), activated N (cm-3) for the four criteria, air density, from the output
    file that has them as time series."""
    for f in ncs:
        with xr.open_dataset(f, decode_times=False) as d:
            if "fa_act" not in d or "height" not in d:
                continue
            fa = d["fa_act"]
            tdim = d["height"].dims[0]
            if tdim not in fa.dims or d.sizes[tdim] < 3:
                continue
            arr = fa.transpose(tdim, ...).values                      # (time, crit, aer) or (time, aer, crit)
            dims = [x for x in fa.dims if x != tdim]
            crit_axis = 1 + next((k for k, x in enumerate(dims) if "crit" in x), 0)
            arr = np.moveaxis(arr, crit_axis, 1)                       # (time, crit, aer)
            n = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]    # index 1 = across all aerosol (nat = 1)
            z = d["height"].values.astype(float)
            z = z - z[0]
            ss = None
            for v in ("ss_spec", "smax", "eq_sat_ratio"):
                if v in d and tdim in d[v].dims:
                    ss = np.asarray(d[v].values, float).reshape(len(z), -1)[:, 0]
                    break
            rho = np.asarray(d["rho"].values, float).reshape(len(z), -1)[:, 0] if "rho" in d and tdim in d["rho"].dims \
                else np.full(len(z), np.nan)
            return pd.DataFrame({"z": z, "ss": ss if ss is not None else np.nan, "rho": rho,
                                 **{f"N{k + 1}": n[:, k] for k in range(min(4, n.shape[1]))}})
    return None


def inspect(a):
    ncs, hour = one_run(a.site, 0, a.w[0], a)
    if not ncs:
        return
    for f in ncs:
        with xr.open_dataset(f, decode_times=False) as d:
            print(f"\n{os.path.basename(f)}: dims {dict(d.sizes)}")
            for v in ("time", "height", "fa_act", "smax", "ss_spec", "rho", "eq_sat_ratio"):
                if v in d:
                    x = d[v]
                    print(f"   {v:12s} {x.dims} {x.shape} first {np.ravel(x.values)[:4]}")
    p = read_profile(ncs)
    print("\nprofile read:" if p is not None else "\nno time-series profile found")
    if p is not None:
        print(p.iloc[np.linspace(0, len(p) - 1, 12).astype(int)].to_string(float_format="{:.4g}".format))


def run(a):
    os.makedirs(OUT, exist_ok=True)
    rows, summ = [], []
    rng = np.random.default_rng(0)
    for site in a.sites:
        with xr.open_dataset(os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")) as ds:
            ok = np.where(np.asarray(ds["thermo_source"].values).astype(str) != "placeholder")[0]
        for ih in rng.choice(ok, min(a.hours, len(ok)), replace=False):
            for w in a.w:
                ncs, hour = one_run(site, int(ih), w, a)
                p = read_profile(ncs) if ncs else None
                if p is None:
                    continue
                p.insert(0, "w", w)
                p.insert(0, "hour", hour)
                p.insert(0, "site", site)
                rows.append(p)
                i150 = int(np.argmin(np.abs(p["z"] - 150)))
                s = dict(site=site, hour=hour, w=w, z_smax=p["z"][int(np.nanargmax(p["ss"]))] if p["ss"].notna().any() else np.nan)
                for k in ("N1", "N3"):
                    for L in LEVELS:
                        j = int(np.argmin(np.abs(p["z"] - L)))
                        s[f"{k}_{L}m_rel"] = p[k][j] / p[k][i150] if p[k][i150] > 0 else np.nan
                summ.append(s)
                print(f"   {site} {hour:%Y-%m-%d %H} w {w:4.2f}: Smax at {s['z_smax']:5.0f} m; N3 25/50/300/600 m vs 150 m "
                      f"{s['N3_25m_rel']:.2f}/{s['N3_50m_rel']:.2f}/{s['N3_300m_rel']:.2f}/{s['N3_600m_rel']:.2f}; "
                      f"N1 25/50 m {s['N1_25m_rel']:.2f}/{s['N1_50m_rel']:.2f}", flush=True)
    for wd in globals().get("_wds", []):
        shutil.rmtree(wd, ignore_errors=True)
    if not rows:
        raise SystemExit("no profiles")
    prof = pd.concat(rows)
    prof.to_csv(os.path.join(OUT, "profiles.csv"), index=False, float_format="%.5g")
    t = pd.DataFrame(summ)
    t.to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.4g")
    print("\nmedian over hours, by updraft (N relative to its value at 150 m):")
    print(t.groupby("w")[["z_smax"] + [f"N3_{L}m_rel" for L in LEVELS] + ["N1_25m_rel", "N1_50m_rel"]]
          .median().to_string(float_format="{:.2f}".format))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = sorted(prof["w"].unique())
    fig, axs = plt.subplots(1, len(ws) + 1, figsize=(4 * (len(ws) + 1), 4.8), sharey=True)
    col = {"N1": "#1f77b4", "N2": "#2ca02c", "N3": "#d62728", "N4": "#9467bd"}
    for ax, w in zip(axs, ws):
        for (s, h), g in prof[prof["w"] == w].groupby(["site", "hour"]):
            j = int(np.argmin(np.abs(g["z"].to_numpy() - 150)))
            for k, c in col.items():
                ref = g[k].to_numpy()[j]
                if ref > 0:
                    ax.plot(g[k] / ref, g["z"], color=c, lw=0.7, alpha=0.6)
        ax.axhline(150, color="k", ls=":", lw=0.8)
        ax.set(xlabel="N / N(150 m)", title=f"w = {w:g} m s$^{{-1}}$", xlim=(0, 1.3))
        ax.grid(alpha=0.3)
    for (s, h, w), g in prof.groupby(["site", "hour", "w"]):
        if g["ss"].notna().any():
            axs[-1].plot(g["ss"], g["z"], lw=0.7, color=plt.get_cmap("viridis")(ws.index(w) / max(1, len(ws) - 1)))
    axs[-1].set(xlabel="supersaturation (PARSEC output)", title="supersaturation, colour = w")
    axs[0].set_ylabel("height above cloud base (m)")
    axs[0].legend(handles=[plt.Line2D([], [], color=c, label=k) for k, c in col.items()], fontsize=8)
    fig.suptitle("Parcel droplet number with height, per volume, relative to 150 m (dotted)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profiles.png"), dpi=150)
    print(f"-> {os.path.relpath(OUT, HERE)}/profiles.csv, summary.csv, profiles.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["inspect", "run"])
    ap.add_argument("--site", default="puijo")
    ap.add_argument("--sites", nargs="*", default=["puijo", "pallas", "lasic", "ena"])
    ap.add_argument("--parsec", default=os.environ.get("PARSEC_DIR",
                    "/nobackup/proj/disk/luftlab/personal/petchu/PARSEC-UFO-stockholm"))
    ap.add_argument("--hours", type=int, default=3, help="random hours per site")
    ap.add_argument("--w", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--cloud-depth", type=float, default=600.0)
    ap.add_argument("--rh0", type=float, default=98.0)
    a = ap.parse_args()
    {"inspect": inspect, "run": run}[a.step](a)


if __name__ == "__main__":
    main()
