#!/usr/bin/env python
"""Zeppelin inputs from the Bolin Centre dataset of Karlsson et al. (2021), doi:10.17043/zeppelin-cloud-aerosol-1.

The observatory (474 m) sits inside the cloud in the Zeppelin cloud hours, so the station data stand in
for cloud-base values. From data/bolin/zeppelin-cloud-aerosol-1_part1.nc4 and _part2.nc4 this writes:

  updrafts/zeppelin_w_cloudbase.nc   per cloud hour: w_mean, sigma_w and the samples w(hour, sample) of
                                     the station's vertical wind (5-7 min means, one per DMPS scan), for
                                     weight_updrafts.py. The 5-7 min averaging removes faster turbulence,
                                     so sigma_w is a lower bound; the mean includes the ridge's forced lift.
  inputs/zeppelin_parcel_inputs.nc   T0 = hourly mean station temperature, P0 = standard-atmosphere
                                     pressure at 474 m (no pressure in the dataset), thermo_source =
                                     "zeppelin_station" (--no-thermo to skip)
  results/zeppelin_activation.nc     per cloud hour: activated fraction AF(D) = cloud residuals (GCVI,
                                     corrected for enrichment and transmission) / whole-air aerosol
                                     (part 2 DMPS, nearest scan within 20 min), the diameter D50 where AF
                                     first reaches 0.5, the corrected residual number and the FM-120
                                     droplet number, for comparison with PARSEC's activation radius.

The residual correction (measured / (enrichment_factor x transmission_efficiency_factor)) is checked by
printing the residual number against the FM-120 droplet number with and without it.

Examples
--------
python zeppelin_bolin.py
python zeppelin_bolin.py --no-thermo
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "data", "bolin", "zeppelin-cloud-aerosol-1_part1.nc4")
P2 = os.path.join(HERE, "data", "bolin", "zeppelin-cloud-aerosol-1_part2.nc4")
ALT = 474.0


def dlog(d):
    ld = np.log10(d)
    e = np.r_[ld[0] - (ld[1] - ld[0]) / 2, (ld[1:] + ld[:-1]) / 2, ld[-1] + (ld[-1] - ld[-2]) / 2]
    return np.diff(e)


def is_cloud(flag):
    v = np.asarray(flag)
    if v.dtype.kind in "SUO":
        return np.array([str(x.decode() if isinstance(x, bytes) else x).strip().lower().startswith("y") for x in v])
    return np.nan_to_num(v.astype(float)) > 0


def d50(af, d):
    """First diameter where AF reaches 0.5 going up in size (log-linear interpolation)."""
    out = np.full(af.shape[0], np.nan)
    for i, a in enumerate(af):
        ok = np.isfinite(a)
        if ok.sum() < 3:
            continue
        aa, dd = a[ok], d[ok]
        k = np.where(aa >= 0.5)[0]
        if len(k) == 0 or k[0] == 0:
            continue
        j = k[0]
        f = (0.5 - aa[j - 1]) / (aa[j] - aa[j - 1])
        out[i] = 10 ** (np.log10(dd[j - 1]) + f * (np.log10(dd[j]) - np.log10(dd[j - 1])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-thermo", action="store_true", help="do not write T0/P0 into the inputs file")
    a = ap.parse_args()

    inp = os.path.join(HERE, "inputs", "zeppelin_parcel_inputs.nc")
    with xr.open_dataset(inp) as ds:
        ds = ds.load()
    hours = pd.DatetimeIndex(ds["hour"].values)
    p1 = xr.open_dataset(P1)
    t1 = pd.DatetimeIndex(p1["time"].values)
    hr1 = t1.floor("h")
    print(f"part 1: {len(t1)} scans {t1.min()} - {t1.max()}; inputs: {len(hours)} cloud hours")

    # ---------------------------------------------------------------- updraft
    w = pd.Series(p1["updraft_velocity"].values.astype(float), index=hr1)
    w = w[np.isfinite(w) & w.index.isin(hours)]
    g = w.groupby(level=0)
    nmax = int(g.size().max()) if len(w) else 1
    W = np.full((len(hours), nmax), np.nan)
    pos = {h: i for i, h in enumerate(hours)}
    for h, s in g:
        W[pos[h], :len(s)] = s.values
    wm, sw, n = g.mean().reindex(hours), g.std().reindex(hours), g.size().reindex(hours).fillna(0)
    os.makedirs(os.path.join(HERE, "updrafts"), exist_ok=True)
    xr.Dataset({"w": (("hour", "sample"), W, {"units": "m s-1"}),
                "w_mean": ("hour", wm.values, {"units": "m s-1"}),
                "sigma_w": ("hour", sw.where(n >= 4).values, {"units": "m s-1",
                            "long_name": "std of 5-7 min mean vertical wind in the hour (lower bound)"}),
                "n_samples": ("hour", n.values.astype(int))},
               coords={"hour": hours.values},
               attrs={"w_source": "zeppelin_station_5min", "source": "Karlsson et al. 2021, Bolin Centre"}
               ).to_netcdf(os.path.join(HERE, "updrafts", "zeppelin_w_cloudbase.nc"))
    print(f"updraft: {int((n > 0).sum())} hours with w, median {int(n[n > 0].median())} samples/h; "
          f"w_mean median {wm.median():.2f}, sigma_w median {sw.median():.2f} m/s, "
          f"positive fraction {np.mean(w > 0):.2f}")

    # ---------------------------------------------------------------- thermodynamics
    T = pd.Series(p1["temperature"].values.astype(float), index=hr1)
    T = T[T.index.isin(hours)].groupby(level=0).mean().reindex(hours) + 273.15
    P0 = 101325.0 * (1 - 2.25577e-5 * ALT) ** 5.25588
    print(f"temperature: {T.notna().sum()} hours, median {T.median() - 273.15:.1f} C; P0 = {P0 / 100:.0f} hPa")
    if not a.no_thermo:
        okT = T.notna().values
        T0, Pa = ds["T0"].values.copy(), ds["P0"].values.copy()
        src = ds["thermo_source"].values.astype(object).copy()
        T0[okT], Pa[okT], src[okT] = T.values[okT], P0, "zeppelin_station"
        ds["T0"] = ("hour", T0, ds["T0"].attrs)
        ds["P0"] = ("hour", Pa, ds["P0"].attrs)
        ds["thermo_source"] = ("hour", src)
        tmp = inp + ".tmp"
        ds.to_netcdf(tmp)
        os.replace(tmp, inp)
        print(f"   -> {os.path.relpath(inp, HERE)}: T0/P0 for {okT.sum()} hours")

    # ---------------------------------------------------------------- activation
    cl = is_cloud(p1["cloud_flag"].values)
    d1 = p1["aerosol_diameters"].values.astype(float) * 1e9
    res = p1["aerosol_particle_size_distribution"].values.astype(float)
    ef = p1["enrichment_factor"].values.astype(float)
    te = p1["transmission_efficiency_factor"].values.astype(float)
    corr = ef * te
    sel = cl & np.isin(hr1, hours) & np.isfinite(corr) & (corr > 0)
    print(f"activation: {cl.sum()} residual scans, {sel.sum()} in the cloud hours with EF x TE")
    res_c = res[sel] / corr[sel, None]

    fm = p1["cloud_particle_size_distribution"].values.astype(float)[sel]
    dc = p1["cloud_diameters"].values.astype(float) * 1e6
    n_fm = np.nansum(fm * dlog(dc), axis=1)
    n_raw = np.nansum(res[sel] * dlog(d1), axis=1)
    n_cor = np.nansum(res_c * dlog(d1), axis=1)
    ok = (n_fm > 1) & (n_raw > 0)
    print(f"   residual number / FM-120 number (median over {ok.sum()} scans): raw {np.median(n_raw[ok] / n_fm[ok]):.2f}, "
          f"/ (EF x TE) {np.median(n_cor[ok] / n_fm[ok]):.2f}  <- should be near 1 if the correction is right")

    p2 = xr.open_dataset(P2)
    t2 = pd.DatetimeIndex(p2["time"].values)
    d2 = p2["aerosol_diameters"].values.astype(float) * 1e9
    wa = p2["aerosol_particle_size_distribution"].values.astype(float)
    ts = t1[sel]
    j = pd.Index(t2).get_indexer(ts, method="nearest", tolerance=pd.Timedelta("20min"))
    whole = np.full((len(ts), len(d1)), np.nan)
    for i, k in enumerate(j):
        if k >= 0:
            r = wa[k]
            m = np.isfinite(r)
            if m.sum() > 3:
                whole[i] = np.interp(np.log10(d1), np.log10(d2[m]), r[m], left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        af = np.where(whole > 1, res_c / whole, np.nan)
    D50 = d50(af, d1)
    print(f"   whole-air match for {np.sum(j >= 0)} scans; D50 found in {np.isfinite(D50).sum()}, "
          f"median {np.nanmedian(D50):.0f} nm (10-90 %: {np.nanpercentile(D50, 10):.0f}-{np.nanpercentile(D50, 90):.0f})")

    df = pd.DataFrame(dict(hour=ts.floor("h"), d50=D50, n_res=n_cor, n_fm=n_fm))
    H = df.groupby("hour").median().reindex(hours)
    AF = pd.DataFrame(af, index=ts.floor("h")).groupby(level=0).median().reindex(hours)
    out = xr.Dataset({"d50": ("hour", H.d50.values, {"units": "nm", "long_name": "diameter where AF reaches 0.5"}),
                      "af": (("hour", "diameter"), AF.to_numpy(float), {"long_name": "activated fraction (hourly median)"}),
                      "n_residual": ("hour", H.n_res.values, {"units": "cm-3", "long_name": "corrected residual number"}),
                      "n_fm120": ("hour", H.n_fm.values, {"units": "cm-3", "long_name": "FM-120 droplet number"}),
                      "n_scans": ("hour", df.groupby("hour").size().reindex(hours).fillna(0).astype(int).values)},
                     coords={"hour": hours.values, "diameter": ("diameter", d1, {"units": "nm"})},
                     attrs={"source": "Karlsson et al. 2021, doi:10.17043/zeppelin-cloud-aerosol-1"})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out.to_netcdf(os.path.join(HERE, "results", "zeppelin_activation.nc"))
    print(f"   -> results/zeppelin_activation.nc: D50 in {out.d50.notnull().sum().item()} of {len(hours)} hours")
    if "Nd_obs" in ds:
        m = np.isfinite(H.n_fm.values) & np.isfinite(ds["Nd_obs"].values)
        if m.any():
            print(f"   our Nd_obs vs FM-120 (hourly medians, {m.sum()} h): ratio median "
                  f"{np.median(ds['Nd_obs'].values[m] / H.n_fm.values[m]):.2f}")


if __name__ == "__main__":
    main()
