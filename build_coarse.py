#!/usr/bin/env python
"""CAMS coarse sea salt and dust per cloud hour, for the optional coarse mode in parsec_run.py (--coarse cams).

The measured spectra stop at 0.45-1 um, so sea salt and dust above that are missing from the parcel runs. This
writes the CAMS (EAC4) coarse-bin masses for every hour of inputs/<site>_parcel_inputs.nc to
inputs/<site>_coarse.nc (a separate file, so the inputs file and the runs made from it are untouched), from the same CAMS
point series and the same 3-hourly > monthly > calendar-month fallback as build_composition.py:

  m_ss_coarse_a   sea salt 0.5-5 um bin (aermr02)          dry mass, ug m-3 (CAMS 80 % RH mass / 4.3)
  m_ss_coarse_b   sea salt 5-20 um bin (aermr03)           dry mass
  m_du_coarse_a   dust 0.55-0.9 um bin (aermr05)           ug m-3
  m_du_coarse_b   dust 0.9-20 um bin (aermr06)             ug m-3
  coarse_source   cams3h / camsmon / camsclim / none

Bin limits are CAMS radii (sea salt at 80 % RH, i.e. about twice the dry radius); parsec_run.py turns them into
dry-diameter ranges (sea salt 0.5-5 / 5-20 um, dust 1.1-1.8 / 1.8-40 um) and number (flat dV/dlogD within a
bin, cut at --coarse-dmax).

python build_coarse.py --sites ena lasic mace-head pallas granada
"""
import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_composition as bc  # noqa: E402

VARS = {"m_ss_coarse_a": ("ss2", 1 / bc.SEASALT_DRY_80RH, "sea salt 0.5-5 um (radius, 80 % RH) bin, dry"),
        "m_ss_coarse_b": ("ss3", 1 / bc.SEASALT_DRY_80RH, "sea salt 5-20 um (radius, 80 % RH) bin, dry"),
        "m_du_coarse_a": ("du2", 1.0, "dust 0.55-0.9 um (radius) bin"),
        "m_du_coarse_b": ("du3", 1.0, "dust 0.9-20 um (radius) bin")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(HERE, "inputs", "*_parcel_inputs.nc")))      # one coarse file per site
    for path in files:
        site = os.path.basename(path)[:-len("_parcel_inputs.nc")]
        if a.sites and site not in a.sites:
            continue
        with xr.open_dataset(path) as d:
            hours = d.indexes["hour"]
        ds = xr.Dataset(coords={"hour": hours.values}, attrs={"site": site, "source": "build_coarse.py, CAMS EAC4"})
        cams = bc.load_cams(site)
        if cams.get("3h") is None and cams.get("mon") is None:
            print(f"{site}: no CAMS files; skipped")
            continue
        res, tag = bc.cams_for_hours(cams, hours)
        for k, (c, f, ln) in VARS.items():
            v = res[c].to_numpy(float) * f if c in res else np.full(len(hours), np.nan)
            ds[k] = ("hour", v, {"units": "ug m-3", "long_name": ln})
        tag = tag.where(tag != "", "none")
        ds["coarse_source"] = ("hour", tag.to_numpy(str))
        out = os.path.join(HERE, "inputs", f"{site}_coarse.nc")
        ds.to_netcdf(out)
        med = {k: np.nanmedian(ds[k].values) for k in VARS}
        print(f"{site:16s} {len(hours)} h ({tag.value_counts().to_dict()}): median dry ss {med['m_ss_coarse_a']:.2f}"
              f" + {med['m_ss_coarse_b']:.2f}, dust {med['m_du_coarse_a']:.2f} + {med['m_du_coarse_b']:.2f} ug m-3")


if __name__ == "__main__":
    main()
