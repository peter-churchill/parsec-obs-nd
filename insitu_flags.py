#!/usr/bin/env python
"""Station visibility, rain and temperature for the in-cloud stations (Puijo, Pallas), from the Zenodo met files.

The Puijo and Pallas cloud hours have a tail with observed Nd < 10 cm-3 where the parcel model gives 30-300:
probably hours when the station is not fully inside cloud (fog edges, broken cloud) or in rain. The met files of
Zenodo 10.5281/zenodo.13358529 (paper_updrafts.py download) give hourly visibility (m), rain intensity (mm h-1)
and temperature; this writes them into results/cloudnet_quality/<site>_flags.csv (visibility_m, rain_mmh,
t_station_C) for nd_ratio_drivers.py, which then bins the ratio by visibility and rain and can screen with
--query, e.g. "visibility_m < 200 and rain_mmh == 0".

python insitu_flags.py
python nd_ratio_drivers.py --sites puijo pallas
python nd_ratio_drivers.py --sites puijo pallas --query "visibility_m < 200 and rain_mmh == 0" --tag _vis200_dry
"""
import netCDF4  # noqa: F401  first, before numpy/stdlib extensions: else Python intermittently aborts (heap corruption) when netCDF4 loads later
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_updrafts as Z  # noqa: E402

FLAGS = os.path.join(HERE, "results", "cloudnet_quality")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", default=["puijo", "pallas"])
    a = ap.parse_args()
    os.makedirs(FLAGS, exist_ok=True)
    for site in a.sites:
        name = f"{site.capitalize()}_meteorological_data_1h_ave.csv"
        if not os.path.exists(os.path.join(Z.ZEN, name)):
            print(f"{site}: no {name} (run paper_updrafts.py download)")
            continue
        m = Z.read_csv(name)
        m = m[~m.index.duplicated()]
        cv, cr, ct = Z.col(m, "vis"), Z.col(m, "rain"), Z.col(m, "temp")
        with xr.open_dataset(os.path.join(HERE, "inputs", f"{site}_parcel_inputs.nc")) as ds:
            hours = pd.DatetimeIndex(ds["hour"].values)
        f = pd.DataFrame({"visibility_m": m[cv], "rain_mmh": m[cr] if cr else np.nan,
                          "t_station_C": m[ct] if ct else np.nan}).reindex(hours)
        p = os.path.join(FLAGS, f"{site}_flags.csv")
        old = pd.read_csv(p, parse_dates=["hour"]).set_index("hour") if os.path.exists(p) else pd.DataFrame(index=hours)
        old = old.drop(columns=[c for c in f.columns if c in old], errors="ignore")
        d = old.join(f, how="outer")
        d.index.name = "hour"
        d.to_csv(p, float_format="%.4g")
        v = f["visibility_m"]
        print(f"{site:8s} met for {v.notna().sum()} of {len(hours)} hours; visibility median {v.median():.0f} m, "
              f"> 200 m in {(v > 200).mean():.0%}, > 1000 m in {(v > 1000).mean():.0%}; "
              f"rain > 0 in {(f['rain_mmh'] > 0).mean():.0%} -> {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
