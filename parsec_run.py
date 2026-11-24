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
    optional extension  --tail lognormal: fitted accumulation-mode continuation above the last measured bin;
                        --coarse cams: CAMS coarse sea salt / dust (build_coarse.py), pure per bin; both on
                        --ext-dlog bins, per-bin composition in custom-composition.dat. N_input then includes
                        them; N_tail_cm3, N_coarse_cm3 and tail_flag are written per run.

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
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
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


# ----------------------------------------------------------------------------- beyond the measured spectrum
# PARSEC column densities (g cm-3), sites.SPECIES order; same values as build_composition.RHO
RHO_SPECIES = np.array([1.841, 1.78, 1.77, 1.50, 2.00, 2.65, 2.165, 1.50])
I_DUST, I_SEASALT = 5, 6
# CAMS coarse bins as dry-diameter ranges (nm): sea salt radius at 80 % RH ~ 2 x dry radius, so radius
# 0.5-5 um -> dry diameter 0.5-5 um; dust radii are dry
COARSE_BINS = {"m_ss_coarse_a": (I_SEASALT, 500.0, 5000.0), "m_ss_coarse_b": (I_SEASALT, 5000.0, 20000.0),
               "m_du_coarse_a": (I_DUST, 1100.0, 1800.0), "m_du_coarse_b": (I_DUST, 1800.0, 40000.0)}


def _lognormal(x, n, mu, s):
    """dN/dlogD of a lognormal mode in log10 D (x, mu) with geometric std s"""
    ls = np.log10(s)
    return n / (np.sqrt(2 * np.pi) * ls) * np.exp(-(x - mu) ** 2 / (2 * ls ** 2))


def fit_tail(dNdlogDp, d_nm, dmin=100.0):
    """Lognormal fitted to the measured bins >= dmin nm, scaled so it meets the mean of the last three measured
    bins (no jump at the edge). Returns (params, scale, flag); params None when the fit is not usable."""
    from scipy.optimize import curve_fit
    y = np.asarray(dNdlogDp, float)
    d = np.asarray(d_nm, float)
    ok = np.isfinite(y)
    i1 = len(ok) - 1 - np.argmax(ok[::-1])
    m = ok & (d >= dmin) & (y > 0)
    if m.sum() < 5:
        return None, 1.0, "few_bins"
    x = np.log10(d)
    try:
        p, _ = curve_fit(_lognormal, x[m], y[m], p0=[np.nanmax(y[m]) * 0.4, np.log10(150), 1.6],
                         bounds=([0, np.log10(50), 1.2], [1e6, np.log10(800), 2.5]), maxfev=4000)
    except Exception:                                                               # noqa: BLE001
        return None, 1.0, "fit_failed"
    last = ok & (np.arange(len(y)) > i1 - 3) & (y > 0)
    f = _lognormal(x[last], *p)
    scale = float(np.exp(np.mean(np.log(np.clip(y[last], 1e-12, None))) - np.mean(np.log(np.clip(f, 1e-12, None)))))
    if not (1 / 3 <= scale <= 3):
        return None, 1.0, "edge_mismatch"
    return p, scale, "ok"


def extend_bins(bins, massfrac, dNdlogDp, d_nm, tail, coarse, dmax_tail, dmax_coarse, dlog):
    """Append bins above the measured spectrum on one log grid (step dlog) from its upper edge:
    tail    lognormal continuation of the measured accumulation mode to dmax_tail nm (fit_tail),
            composition = the hour's bulk mass fractions
    coarse  dict of CAMS coarse masses (ug m-3, COARSE_BINS keys): within each CAMS bin dV/dlogD is flat;
            only the part above the measured edge and below dmax_coarse is used; pure sea salt / dust
    Bins with both get the mass-weighted composition. Returns (bins, per-bin mass fractions, info)."""
    r_mid, r_lo, r_hi, n = bins
    mf = np.asarray(massfrac, float) / np.sum(massfrac)
    comp = np.tile(mf, (len(n), 1))
    info = dict(N_tail_cm3=0.0, N_coarse_cm3=0.0, tail_flag="off")
    top = np.log10(2e9 * r_hi[-1])                      # measured upper edge, log10 nm
    end = np.log10(max(dmax_tail if tail else 0, dmax_coarse if coarse else 0))
    if end <= top + dlog / 2:
        if tail:
            info["tail_flag"] = "spectrum_reaches_max"
        return bins, comp, info
    e = np.arange(top, end + 1e-9, dlog)
    if e[-1] < end - 1e-6:
        e = np.append(e, end)
    lo, hi = e[:-1], e[1:]
    mid = 0.5 * (lo + hi)
    dm = 10 ** mid * 1e-9                                # m
    vol1 = np.pi / 6 * dm ** 3                           # m3 per particle
    num = np.zeros(len(mid))                             # m-3
    mass = np.zeros((len(mid), len(mf)))                 # g m-3 per species
    if tail:
        p, scale, flag = fit_tail(dNdlogDp, d_nm)
        info["tail_flag"] = flag
        if p is not None:
            use = hi <= np.log10(dmax_tail) + 1e-9
            nt = np.where(use, scale * _lognormal(mid, *p) * (hi - lo) * 1e6, 0.0)
            num += nt
            rho_b = 1.0 / np.sum(mf / RHO_SPECIES) * 1e6               # g m-3 of particle
            mass += (nt * vol1 * rho_b)[:, None] * mf[None, :]
            info["N_tail_cm3"] = float(nt.sum() * 1e-6)
    if coarse:
        for k, (isp, d0, d1) in COARSE_BINS.items():
            mtot = float(coarse.get(k, np.nan))
            if not (np.isfinite(mtot) and mtot > 0):
                continue
            l0, l1 = np.log10(d0), np.log10(d1)
            ov = np.clip(np.minimum(hi, l1) - np.maximum(lo, l0), 0, None)   # overlap of each new bin with the CAMS bin
            ov[hi > np.log10(dmax_coarse) + 1e-9] = 0
            if ov.sum() == 0:
                continue
            mb = mtot * 1e-6 * ov / (l1 - l0)                                # g m-3 per new bin (flat dV/dlogD)
            nb = mb / (RHO_SPECIES[isp] * 1e6 * vol1)
            num += nb
            mass[:, isp] += mb
            info["N_coarse_cm3"] += float(nb.sum() * 1e-6)
    keep = num > 0
    if not keep.any():
        return bins, comp, info
    tot = mass[keep].sum(axis=1, keepdims=True)
    new_comp = np.where(tot > 0, mass[keep] / np.where(tot > 0, tot, 1), mf[None, :])
    rr = lambda lg: 0.5 * 10.0 ** lg * 1e-9               # noqa: E731
    out = (np.concatenate([r_mid, rr(mid[keep])]), np.concatenate([r_lo, rr(lo[keep])]),
           np.concatenate([r_hi, rr(hi[keep])]), np.concatenate([n, num[keep]]))
    return out, np.vstack([comp, new_comp]), info


# ----------------------------------------------------------------------------- namelists and .dat files
def _set(text, key, value):
    """Replace `key = <value>,` at the start of a namelist line (keeps the comment)."""
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)[^,!\n]*", re.M)
    if not pat.search(text):
        raise KeyError(f"namelist key '{key}' not found in template")
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def write_inputs(wd, tmpl, bins, massfrac, T0, P0, w, rh0, depth, dt, dtsub, entrain=None):
    """entrain: None (adiabatic, the default) or dict(emur, radius, lapse): pseudo-adiabatic ascent mixing in
    environmental air at rate emur / radius per metre (Lee & Pruppacher 1977); environment temperature T0 - lapse*z
    (K km-1) written per run, environment humidity PARSEC's built-in profile (~95 % RH near the start)."""
    inp = os.path.join(wd, "inputs")
    os.makedirs(inp, exist_ok=True)
    for f in ("namelist_developer.in", "environmental_temperature_profile_user.dat",
              "rebin-midpoint_radii.dat", "rebin-size_limits.dat", "wavelengths.dat"):
        shutil.copy(os.path.join(tmpl, f), os.path.join(inp, f))

    sw = open(os.path.join(tmpl, "namelist_switches.in")).read()
    for k, v in dict(SWITCHES, **({"iupadiabat": "F"} if entrain else {})).items():
        sw = _set(sw, k, v)
    if entrain:
        z = np.arange(0.0, depth + 1000.0, 10.0)
        np.savetxt(os.path.join(inp, "environmental_temperature_profile_user.dat"),
                   np.column_stack([z, T0 - entrain["lapse"] * 1e-3 * z]), fmt="%.1f %.3f")
    open(os.path.join(inp, "namelist_switches.in"), "w").write(sw)

    r_mid, r_lo, r_hi, n = bins
    tlim = (60.0 + depth) / w * 1.5 + 120.0
    uv = open(os.path.join(tmpl, "namelist_UserVars.in")).read()
    for k, v in dict(dt=dt, dtsub=dtsub, time_limit=f"{tlim:.0f}", na=len(n), nat=1,
                     initial_temperature=f"{T0:.3f}", initial_pressure=f"{P0:.1f}", updraft=f"{w:g}",
                     downdraft=0, initial_relhum=f"{rh0:g}",
                     entrainment_param=f"{entrain['emur']:g}" if entrain else 0,
                     cloud_depth=f"{depth:g}", initial_height=0).items():
        uv = _set(uv, k, v)
    if entrain:
        uv = _set(uv, "initial_radius", f"{entrain['radius']:g}")
    open(os.path.join(inp, "namelist_UserVars.in"), "w").write(uv)

    np.savetxt(os.path.join(inp, "custom-midpoint_radii.dat"), r_mid, fmt="%.8e")
    np.savetxt(os.path.join(inp, "custom-size_limits.dat"), np.column_stack([r_lo, r_hi]), fmt="%.8e")
    np.savetxt(os.path.join(inp, "custom-concentration.dat"), n, fmt="%.8e")
    mf = np.asarray(massfrac, float)
    mf = np.tile(mf / mf.sum(), (len(n), 1)) if mf.ndim == 1 else mf / mf.sum(axis=1, keepdims=True)
    np.savetxt(os.path.join(inp, "custom-composition.dat"), mf, fmt="%.10f", delimiter="\t")
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
    ap.add_argument("--tail", choices=["none", "lognormal"], default="none",
                    help="continue the measured spectrum above its last bin with a fitted lognormal")
    ap.add_argument("--tail-dmax", type=float, default=3000.0, help="nm; upper end of the tail")
    ap.add_argument("--coarse", choices=["none", "cams"], default="none",
                    help="add CAMS coarse sea salt and dust above the measured spectrum (inputs/<site>_coarse.nc "
                         "from build_coarse.py)")
    ap.add_argument("--coarse-dmax", type=float, default=10000.0, help="nm; largest coarse particles used")
    ap.add_argument("--ext-dlog", type=float, default=0.05, help="log10 width of the added bins")
    ap.add_argument("--entrainment", type=float, default=None,
                    help="PARSEC emur (0-0.6): pseudo-adiabatic ascent with entrainment of environmental air at "
                         "emur / --parcel-radius per metre (default: adiabatic, no entrainment)")
    ap.add_argument("--parcel-radius", type=float, default=350.0, help="m, with --entrainment")
    ap.add_argument("--env-lapse", type=float, default=6.5, help="K km-1 environmental lapse rate, with --entrainment")
    ap.add_argument("--subsample", type=int, default=1,
                    help="run every n-th cloud hour only (sensitivity tests); chunks are made after subsampling")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--dtsub", type=float, default=0.1)
    ap.add_argument("--unit-tol", type=float, default=0.05)
    ap.add_argument("--keep-workdir", action="store_true", help="debug: keep the scratch dir")
    a = ap.parse_args()

    ds = xr.open_dataset(a.inputs)
    sel = np.arange(0, ds.sizes["hour"], max(1, a.subsample))
    nh = len(sel)
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
    coarse_ds = None
    if a.coarse == "cams":
        cp = os.path.join(os.path.dirname(os.path.abspath(a.inputs)), f"{site}_coarse.nc")
        if not os.path.exists(cp):
            sys.exit(f"--coarse cams: {cp} missing (run build_coarse.py --sites {site})")
        coarse_ds = xr.open_dataset(cp).load()
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
        for ih in sel[h0:h1]:
            hour = pd.Timestamp(ds["hour"].values[ih])
            spec = ds["dNdlogDp"].values[ih]
            bins = spectrum_to_bins(spec, d_nm, a.subdivide)
            mfrac = ds["massfrac"].values[ih]
            ext = dict(N_tail_cm3=0.0, N_coarse_cm3=0.0, tail_flag="off")
            if a.tail != "none" or coarse_ds is not None:
                cm = None
                if coarse_ds is not None:
                    c = coarse_ds.sel(hour=ds["hour"].values[ih])
                    cm = {k: float(c[k]) for k in COARSE_BINS if k in c}
                bins, mfrac, ext = extend_bins(bins, mfrac, spec, d_nm, a.tail != "none", cm,
                                               a.tail_dmax, a.coarse_dmax, a.ext_dlog)
            n_in_cm3 = bins[3].sum() * 1e-6
            T0, P0 = float(ds["T0"].values[ih]), float(ds["P0"].values[ih])
            rho0 = P0 / (RD * T0) * 1e3                                   # g m-3 (dry; ~0.5% low)
            for w in a.w_grid:
                ent = None if a.entrainment is None else dict(emur=a.entrainment, radius=a.parcel_radius,
                                                              lapse=a.env_lapse)
                tlim = write_inputs(wd, tmpl, bins, mfrac, T0, P0, w,
                                    a.rh0, a.cloud_depth, a.dt, a.dtsub, entrain=ent)
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
                                 N_input_cm3=n_in_cm3, nbins=len(bins[3]), T0=T0, P0=P0, **ext,
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
