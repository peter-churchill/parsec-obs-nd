#!/usr/bin/env python
"""Build PARSEC parcel-model inputs for the screened cloud hours of each site.

One NetCDF per site, inputs/<site>_parcel_inputs.nc, with dims
(hour, diameter, species):

    dNdlogDp      measured hourly size distribution on pnsd_tools.GRID_NM (cm-3)
    massfrac      dry mass fractions in PARSEC's species order (sites.SPECIES)
    T0, P0        parcel start temperature (K) and pressure (Pa)
    Nd_obs, N100  observed Nd and N(>100 nm) for the comparison (cm-3)
    comp_source, thermo_source   where massfrac and T0/P0 came from

The cloud hours are those that enter the susceptibility fits (pnsd_tools.nd_hours:
cloud_frac >= 0.5, cloud base <= 2000 m, top <= 3000 m), plus the NDROP error
screen (Nd error / Nd <= 0.5) for NDROP sites, so the parcel runs line up with
the synthesis table. Hours whose spectrum does not reach down to --dmin-max or
up to --dmax-min are dropped and counted.

Composition and T0/P0 are filled with clearly flagged placeholders here; the
CAMS/ACSM and thermodynamics stages overwrite them in place and change the
*_source flags. The PARSEC runs read whatever is in the file at run time.

Examples
--------
python build_parcel_inputs.py --root /path/to/obs-ccn-nd-susceptibility --sites ena hyytiala puijo
export OBS_REPO=/path/to/obs-ccn-nd-susceptibility; python build_parcel_inputs.py --all
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
OBS_REPO = os.environ.get("OBS_REPO")      # the observation repo (built files, results/pnsd_cache, pnsd_tools.py)
sys.path.insert(0, HERE)

from sites import (PLACEHOLDER_P0_PA, PLACEHOLDER_T0_K, REGIME_COMPOSITION,  # noqa: E402
                   SITES, SPECIES)


def load_pnsd(spec, root):
    """Hourly dN/dlogDp (cm-3) on GRID_NM as a DataFrame indexed by hour."""
    kind, rel = spec
    path = os.path.join(root, rel)
    if kind == "cache":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            return pd.DataFrame(ds["dNdlogDp"].values, index=pd.DatetimeIndex(ds["time"].values),
                                columns=ds["diameter"].values)
    if kind == "csv":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        sys.path.insert(0, root)
        import pnsd_tools as pt          # from the observation repo; only to grid the Virtanen CSVs (cached after first use)
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        cache = os.path.join(root, "results", "pnsd_cache", f"{stem}_hourly.nc")
        return pt.load_or_build(cache, builder=lambda: pt.hourly_pnsd_csv(path))
    raise ValueError(f"unknown pnsd spec {spec}")


def _to_kelvin(x, units):
    if not np.isfinite(x).any():
        return x
    u = units.lower()
    if u in ("degc", "c", "celsius", "degree_celsius", "deg c") or (u == "" and np.nanmedian(x) < 100):
        return x + 273.15
    return x


def _to_pa(x, units):
    if not np.isfinite(x).any():
        return x
    u = units.lower()
    if u == "kpa" or (u == "" and np.nanmedian(x) < 200):
        return x * 1000.0
    if u in ("hpa", "mb", "mbar") or (u == "" and np.nanmedian(x) < 2000):
        return x * 100.0
    return x


def screened_hours(ds, kind, max_rel_err=0.5):
    """Mask of built-file hours used in the susceptibility fits."""
    t = pd.DatetimeIndex(ds["time"].values)
    nd = ds["CDNC"].values
    m = np.isfinite(nd) & (nd > 0)
    if "cloud_frac" in ds:
        m &= ds["cloud_frac"].values >= 0.5
    if kind == "cloudnet":                       # base/top screens are Cloudnet-only in the quicklooks
        if "cloud_base_height" in ds:
            m &= ~(ds["cloud_base_height"].values > 2000.0)
        if "cloud_top_height" in ds:
            m &= ~(ds["cloud_top_height"].values > 3000.0)
    if kind == "ndrop" and "drop_number_conc_toterror" in ds:
        rel = ds["drop_number_conc_toterror"].values / nd
        m &= np.isfinite(rel) & (rel <= max_rel_err)
    return t, m


def spectrum_ok(row, d, dmin_max, dmax_min, max_gap_frac=0.1):
    """Contiguous measured span reaching <= dmin_max and >= dmax_min, few interior gaps."""
    fin = np.isfinite(row)
    if fin.sum() < 5:
        return False
    i0, i1 = np.argmax(fin), len(fin) - 1 - np.argmax(fin[::-1])
    if d[i0] > dmin_max or d[i1] < dmax_min:
        return False
    return (~fin[i0:i1 + 1]).mean() <= max_gap_frac


def build_site(name, root, outdir, dmin_max, dmax_min):
    cfg = SITES[name]
    built = os.path.join(root, cfg["built"])
    if not os.path.exists(built):
        print(f"{name}: SKIP, built file missing ({cfg['built']})")
        return None
    try:
        pnsd = load_pnsd(cfg["pnsd"], root)
    except FileNotFoundError as e:
        print(f"{name}: SKIP, size-distribution source missing ({e})")
        return None

    with xr.open_dataset(built) as ds:
        t, m = screened_hours(ds, cfg["kind"])
        i100 = int(np.argmin(np.abs(ds["diameter"].values - 100.0)))
        nd_obs = ds["CDNC"].values[m]
        n100 = ds["CCN"].values[m, i100]
        cbh = ds["cloud_base_height"].values[m] if "cloud_base_height" in ds else np.full(m.sum(), np.nan)
        # measured cloud-base T and P (ARM NDROP builds); converted to K and Pa
        cbt = ds["cloud_base_temperature"].values[m].astype(float) if "cloud_base_temperature" in ds else np.full(m.sum(), np.nan)
        cbp = ds["cloud_base_pressure"].values[m].astype(float) if "cloud_base_pressure" in ds else np.full(m.sum(), np.nan)
        t_units = str(ds["cloud_base_temperature"].attrs.get("units", "")) if "cloud_base_temperature" in ds else ""
        p_units = str(ds["cloud_base_pressure"].attrs.get("units", "")) if "cloud_base_pressure" in ds else ""
        site_attr = str(ds.attrs.get("site", name))
    cbt, cbp = _to_kelvin(cbt, t_units), _to_pa(cbp, p_units)
    hours = t[m]
    n_screen = len(hours)

    d = pnsd.columns.to_numpy(dtype=float)
    spec = pnsd.reindex(hours).to_numpy(dtype=float)
    has = np.isfinite(spec).any(axis=1)
    ok = np.array([h and spectrum_ok(r, d, dmin_max, dmax_min) for h, r in zip(has, spec)], bool)
    print(f"{name}: {n_screen} screened cloud hours, {has.sum()} with a spectrum, "
          f"{ok.sum()} pass the {dmin_max:.0f}-{dmax_min:.0f} nm coverage check")
    if not ok.any():
        return None

    spec, hours = spec[ok], hours[ok]
    nh = len(hours)
    comp = np.tile(np.asarray(REGIME_COMPOSITION[cfg["regime"]], float), (nh, 1))
    cbt, cbp = cbt[ok], cbp[ok]
    have_tp = np.isfinite(cbt) & np.isfinite(cbp) & (cbt > 200) & (cbt < 320) & (cbp > 50000) & (cbp < 105000)
    T0 = np.where(have_tp, cbt, PLACEHOLDER_T0_K)
    P0 = np.where(have_tp, cbp, PLACEHOLDER_P0_PA)
    tsrc = np.where(have_tp, "built_file_cloud_base", "placeholder").astype(object)
    if have_tp.any():
        print(f"   cloud-base T/P from the built file for {have_tp.sum()} of {nh} hours "
              f"(T median {np.median(cbt[have_tp]):.1f} K, P median {np.median(cbp[have_tp]) / 100:.0f} hPa)")
    ds_out = xr.Dataset(
        {"dNdlogDp": (("hour", "diameter"), spec, {"units": "cm-3", "long_name": "hourly dN/dlogDp (dry)"}),
         "massfrac": (("hour", "species"), comp, {"units": "1", "long_name": "dry mass fractions, PARSEC order"}),
         "comp_source": ("hour", np.array(["regime_placeholder"] * nh, dtype=object)),
         "T0": ("hour", T0, {"units": "K", "long_name": "parcel start temperature"}),
         "P0": ("hour", P0, {"units": "Pa", "long_name": "parcel start pressure"}),
         "thermo_source": ("hour", tsrc),
         "Nd_obs": ("hour", nd_obs[ok], {"units": "cm-3"}),
         "N100": ("hour", n100[ok], {"units": "cm-3"}),
         "cloud_base_height": ("hour", cbh[ok], {"units": "m"})},
        coords={"hour": hours.values,
                "diameter": ("diameter", d, {"units": "nm"}),
                "species": ("species", SPECIES)},
        attrs={"site": name, "site_long": site_attr, "kind": cfg["kind"], "regime": cfg["regime"],
               "built_file": cfg["built"], "pnsd_source": f"{cfg['pnsd'][0]}:{cfg['pnsd'][1]}",
               "screens": "cloud_frac>=0.5; base<=2000 m; top<=3000 m; NDROP err/Nd<=0.5",
               "coverage": f"spectrum reaches <= {dmin_max} nm and >= {dmax_min} nm",
               "n_screened_hours": n_screen,
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}_parcel_inputs.nc")
    if os.path.exists(out):
        os.remove(out)
    ds_out.to_netcdf(out, encoding={"dNdlogDp": {"zlib": True, "complevel": 4}})
    print(f"   -> {os.path.relpath(out, root)} ({nh} hours)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sites", nargs="+", choices=sorted(SITES))
    g.add_argument("--all", action="store_true")
    ap.add_argument("--root", default=OBS_REPO,
                    help="observation repo root (default: $OBS_REPO), holding results/ and pnsd_tools.py")
    ap.add_argument("--outdir", default=os.path.join(HERE, "inputs"))
    ap.add_argument("--dmin-max", type=float, default=20.0, help="spectrum must start at or below this (nm)")
    ap.add_argument("--dmax-min", type=float, default=300.0, help="spectrum must reach at least this (nm)")
    a = ap.parse_args()
    if not a.root:
        ap.error("give --root or set OBS_REPO to the observation repo (the one with results/ and pnsd_tools.py)")
    for s in (sorted(SITES) if a.all else a.sites):
        build_site(s, a.root, a.outdir, a.dmin_max, a.dmax_min)


if __name__ == "__main__":
    main()
