#!/usr/bin/env python
"""Run PARSEC-UFO for a chunk of cloud hours on a fixed updraft grid.

Each hour of inputs/<site>_parcel_inputs.nc is run at every updraft in
--w-grid (default 10 values, 0.05-3 m/s). Hour-specific updraft PDFs are applied
afterwards by weight_updrafts.py, which interpolates Nd(w) over the measured
w distribution, so these runs never have to be repeated when the updraft
data or its screening changes.

PARSEC setup per run (everything else from the template namelists):
    customaer = 2       measured spectrum + composition from .dat files, nat = 1 (internal mixture)
    iupadiabat = T      adiabatic ascent, iupconst = 1 constant updraft
    kprecip = 0         no collision-coalescence; aqueous chemistry, SVOCs, compressed film off
    start               T0, P0 from the inputs file, RH = --rh0, then --cloud-depth m above cloud base
    concentrations      written in m-3 (PARSEC divides by air density in g m-3; checked below)

Every run is done in a node-local scratch directory ($TMPDIR, or /tmp), the
needed numbers are parsed from logs/logs.txt and the run's NetCDF output is
deleted, so 10^5 runs leave one CSV per chunk (runs/<site>/chunk_*.csv).

Unit check: PARSEC's reported initial number, scaled back from end-of-run to
start air density, must match the integral of the input spectrum within
--unit-tol; otherwise the run is flagged "unit_mismatch".

Examples
--------
# how many array tasks a site needs at 25 hours per task
python parsec_run.py --inputs inputs/ena_parcel_inputs.nc --chunk-hours 25 --print-ntasks
# one chunk (normally SLURM_ARRAY_TASK_ID picks it)
python parsec_run.py --inputs inputs/ena_parcel_inputs.nc --parsec ~/PARSEC-UFO-stockholm \\
    --chunk-hours 25 --task 0 --outdir runs
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import xarray as xr

W_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]   # m s-1
RD = 287.05                                                      # J kg-1 K-1

SWITCHES = dict(initflag=1, customaer=2, initaer=1, output_logs="T", iupadiabat="T", iupconst=1,
                kprecip=0, vcycles=0, irainfall="F", icwc="F", icondsvocs="F", icompfilm="F",
                iendssmax="F", istopheight="F", interp="F", ihoutputs="F", out_aerdat="F",
                initwav=1, initrebin=1, envprofile=1)


# ----------------------------------------------------------------------------- spectrum -> PARSEC bins
def spectrum_to_bins(dNdlogDp, d_nm, subdivide=2):
    """Measured dN/dlogDp (cm-3) on centres d_nm -> PARSEC bins.

    Uses the contiguous measured span, fills interior gaps by interpolation in log D,
    treats dN/dlogDp as constant within each grid bin and splits each bin into
    `subdivide` equal log-width sub-bins.

    Returns (r_mid_m, r_lo_m, r_hi_m, n_m3); n_m3 is number per m3 in each bin.
    """
    y = np.asarray(dNdlogDp, float).copy()
    fin = np.isfinite(y)
    i0, i1 = np.argmax(fin), len(fin) - 1 - np.argmax(fin[::-1])
    x = np.log10(np.asarray(d_nm, float))
    y, x = y[i0:i1 + 1], x[i0:i1 + 1]
    gap = ~np.isfinite(y)
    if gap.any():
        y[gap] = np.interp(x[gap], x[~gap], y[~gap])
    y = np.clip(y, 0.0, None)

    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[1:] + x[:-1])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    s = int(max(1, subdivide))
    sub = np.concatenate([np.linspace(edges[k], edges[k + 1], s + 1)[:-1] for k in range(len(x))] + [edges[-1:]])
    lo, hi = sub[:-1], sub[1:]
    dens = np.repeat(y, s)                                # dN/dlogDp per sub-bin, cm-3
    n_m3 = dens * (hi - lo) * 1e6                         # number per m3 in each sub-bin
    r = lambda logd: 0.5 * 10.0 ** logd * 1e-9            # noqa: E731  log10 D (nm) -> radius (m)
    return r(0.5 * (lo + hi)), r(lo), r(hi), n_m3


# ----------------------------------------------------------------------------- namelists and .dat files
def _set(text, key, value):
    """Replace `key = <value>,` at the start of a namelist line (keeps the comment)."""
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)[^,!\n]*", re.M)
    if not pat.search(text):
        raise KeyError(f"namelist key '{key}' not found in template")
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def write_inputs(wd, tmpl, bins, massfrac, T0, P0, w, rh0, depth, dt, dtsub):
    inp = os.path.join(wd, "inputs")
    os.makedirs(inp, exist_ok=True)
    for f in ("namelist_developer.in", "environmental_temperature_profile_user.dat",
              "rebin-midpoint_radii.dat", "rebin-size_limits.dat", "wavelengths.dat"):
        shutil.copy(os.path.join(tmpl, f), os.path.join(inp, f))

    sw = open(os.path.join(tmpl, "namelist_switches.in")).read()
    for k, v in SWITCHES.items():
        sw = _set(sw, k, v)
    open(os.path.join(inp, "namelist_switches.in"), "w").write(sw)

    r_mid, r_lo, r_hi, n = bins
    tlim = (60.0 + depth) / w * 1.5 + 120.0
    uv = open(os.path.join(tmpl, "namelist_UserVars.in")).read()
    for k, v in dict(dt=dt, dtsub=dtsub, time_limit=f"{tlim:.0f}", na=len(n), nat=1,
                     initial_temperature=f"{T0:.3f}", initial_pressure=f"{P0:.1f}", updraft=f"{w:g}",
                     downdraft=0, initial_relhum=f"{rh0:g}", entrainment_param=0,
                     cloud_depth=f"{depth:g}", initial_height=0).items():
        uv = _set(uv, k, v)
    open(os.path.join(inp, "namelist_UserVars.in"), "w").write(uv)

    np.savetxt(os.path.join(inp, "custom-midpoint_radii.dat"), r_mid, fmt="%.8e")
    np.savetxt(os.path.join(inp, "custom-size_limits.dat"), np.column_stack([r_lo, r_hi]), fmt="%.8e")
    np.savetxt(os.path.join(inp, "custom-concentration.dat"), n, fmt="%.8e")
    mf = np.asarray(massfrac, float) / np.sum(massfrac)
    np.savetxt(os.path.join(inp, "custom-composition.dat"), np.tile(mf, (len(n), 1)), fmt="%.10f", delimiter="\t")
    return tlim


# ----------------------------------------------------------------------------- log parsing
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN"


def _nums(line):
    return [float(v) for v in re.findall(_NUM, line.split(":", 1)[1])] if ":" in line else []


def parse_log(path, nat=1):
    """Pull the end-of-run numbers out of PARSEC's logs.txt (column `nat` = across all aerosols)."""
    out = dict(complete=False)
    if not os.path.exists(path):
        return out
    txt = open(path, errors="replace").read()
    out["complete"] = "Simulation Complete" in txt
    q = txt.split("Simulation output quicklook")[-1] if "Simulation output quicklook" in txt else ""
    for line in q.splitlines():
        s = line.strip()
        v = _nums(s)
        if not v:
            continue
        for k in (1, 2, 3, 4):
            if s.startswith(f"Activated fraction [criteria {k}]"):
                out[f"AF_c{k}"] = v[nat] if len(v) > nat else v[0]
        if s.startswith("Initial aerosols") and "N_init_end" not in out:
            out["N_init_end"] = v[nat] if len(v) > nat else v[0]
        elif s.startswith("Activated aerosols") and "N_act_end" not in out:
            out["N_act_end"] = v[nat] if len(v) > nat else v[0]
        elif s.startswith("Min dry activation radius"):
            out["r_act_min_nm"] = v[0]
        elif s.startswith("Maximum supersaturation"):
            out["Smax_pct"] = v[-1]
        elif s.startswith("Air density"):
            out["rho_end_gm3"] = v[0]
        elif s.startswith("Cloud base"):
            out["cloud_base_m"] = v[0]
        elif s.startswith("Mean droplet radius") and "r_mean_um" not in out:
            out["r_mean_um"] = v[nat] if len(v) > nat else v[0]
        elif s.startswith("Altitude above cloud base"):
            out["z_above_base_m"] = v[0]
    return out


# ----------------------------------------------------------------------------- driver
def run_one(exe, wd, timeout):
    for d in ("outputs", "logs"):
        shutil.rmtree(os.path.join(wd, d), ignore_errors=True)
        os.makedirs(os.path.join(wd, d))
    env = dict(os.environ)
    if os.environ.get("PARSEC_LD_LIBRARY_PATH"):            # module libraries for the binary only (see sbatch)
        env["LD_LIBRARY_PATH"] = os.environ["PARSEC_LD_LIBRARY_PATH"]
    t0 = time.time()
    try:
        p = subprocess.run([exe], cwd=wd, capture_output=True, text=True, timeout=timeout, env=env)
        rc, err = p.returncode, (p.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        rc, err = -9, "timeout"
    return rc, err, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="inputs/<site>_parcel_inputs.nc")
    ap.add_argument("--parsec", help="PARSEC-UFO directory (contains parsec-ufo and inputs/)")
    ap.add_argument("--exe", help="PARSEC binary (default <parsec>/parsec-ufo)")
    ap.add_argument("--templates", help="template inputs dir (default <parsec>/inputs)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    ap.add_argument("--chunk-hours", type=int, default=25)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--print-ntasks", action="store_true", help="print the number of chunks and exit")
    ap.add_argument("--max-hours", type=int, help="testing: only the first N hours of the chunk")
    ap.add_argument("--w-grid", type=float, nargs="+", default=W_GRID)
    ap.add_argument("--rh0", type=float, default=98.0, help="start relative humidity (%%)")
    ap.add_argument("--cloud-depth", type=float, default=150.0, help="m above cloud base where the run ends")
    ap.add_argument("--subdivide", type=int, default=2, help="sub-bins per measured grid bin")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--dtsub", type=float, default=0.1)
    ap.add_argument("--unit-tol", type=float, default=0.05)
    ap.add_argument("--keep-workdir", action="store_true", help="debug: keep the scratch dir")
    a = ap.parse_args()

    ds = xr.open_dataset(a.inputs)
    nh = ds.sizes["hour"]
    ntask = math.ceil(nh / a.chunk_hours)
    if a.print_ntasks:
        print(ntask)
        return
    if a.task >= ntask:
        print(f"task {a.task} >= {ntask} chunks; nothing to do")
        return
    if not a.parsec and not (a.exe and a.templates):
        sys.exit("give --parsec, or both --exe and --templates")
    exe = os.path.abspath(a.exe or os.path.join(a.parsec, "parsec-ufo"))
    tmpl = os.path.abspath(a.templates or os.path.join(a.parsec, "inputs"))

    site = ds.attrs["site"]
    h0, h1 = a.task * a.chunk_hours, min(nh, (a.task + 1) * a.chunk_hours)
    if a.max_hours:
        h1 = min(h1, h0 + a.max_hours)
    d_nm = ds["diameter"].values
    outdir = os.path.join(a.outdir, site)
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"chunk_{a.task:05d}.csv")
    wd = tempfile.mkdtemp(prefix=f"parsec_{site}_{a.task}_", dir=os.environ.get("TMPDIR", "/tmp"))
    print(f"{site}: hours {h0}-{h1 - 1} of {nh}, {len(a.w_grid)} updrafts each, scratch {wd}", flush=True)

    rows = []
    try:
        for ih in range(h0, h1):
            hour = pd.Timestamp(ds["hour"].values[ih])
            bins = spectrum_to_bins(ds["dNdlogDp"].values[ih], d_nm, a.subdivide)
            n_in_cm3 = bins[3].sum() * 1e-6
            T0, P0 = float(ds["T0"].values[ih]), float(ds["P0"].values[ih])
            rho0 = P0 / (RD * T0) * 1e3                                   # g m-3 (dry; ~0.5% low)
            for w in a.w_grid:
                tlim = write_inputs(wd, tmpl, bins, ds["massfrac"].values[ih], T0, P0, w,
                                    a.rh0, a.cloud_depth, a.dt, a.dtsub)
                rc, err, wall = run_one(exe, wd, timeout=max(600, 20 * tlim))
                r = parse_log(os.path.join(wd, "logs", "logs.txt"))
                status = "ok"
                if rc != 0 or not r.get("complete"):
                    status = f"failed rc={rc}"
                elif "N_init_end" in r and "rho_end_gm3" in r:
                    n_back = r["N_init_end"] * rho0 / r["rho_end_gm3"]
                    if abs(n_back / n_in_cm3 - 1) > a.unit_tol:
                        status = f"unit_mismatch ({n_back:.1f} vs {n_in_cm3:.1f})"
                rows.append(dict(site=site, hour_index=ih, hour=hour, w=w, status=status,
                                 N_input_cm3=n_in_cm3, nbins=len(bins[3]), T0=T0, P0=P0,
                                 comp_source=str(ds["comp_source"].values[ih]),
                                 thermo_source=str(ds["thermo_source"].values[ih]),
                                 wall_s=round(wall, 2), err=err.replace("\n", " ") if status != "ok" else "",
                                 **{k: v for k, v in r.items() if k != "complete"}))
            last = rows[-len(a.w_grid):]
            af = [x.get("AF_c3", np.nan) for x in last]
            print(f"  {hour:%Y-%m-%d %H:%M}  N={n_in_cm3:7.1f} cm-3  AF(c3) {np.nanmin(af):.2f}-{np.nanmax(af):.2f}"
                  f"  {sum(x['wall_s'] for x in last):5.1f} s  {sum(x['status'] != 'ok' for x in last)} bad", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)          # rewrite as we go: a killed job keeps its hours
    finally:
        if not a.keep_workdir:
            shutil.rmtree(wd, ignore_errors=True)
    df = pd.DataFrame(rows)
    if len(df):
        for k in (1, 2, 3, 4):
            if f"AF_c{k}" in df:
                df[f"Nd_c{k}"] = df[f"AF_c{k}"] * df["N_input_cm3"]
        df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} runs, {(df['status'] != 'ok').sum() if len(df) else 0} not ok")


if __name__ == "__main__":
    main()
