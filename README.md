# parsec-obs-nd

PARSEC parcel-model droplet number for every screened cloud hour of the ground
observation sites, for comparison with the observed Nd and the observed CCN–Nd
susceptibility. Uses the Stockholm fork of PARSEC-UFO with `customaer = 2`
(measured spectrum and composition read bin by bin, no lognormal fit).

Two other directories are needed:

- **the observation repo** (`obs-ccn-nd-susceptibility`, locally `ARM_Data`): built
  hourly Nd files, `results/pnsd_cache/`, `pnsd_tools.py`. Pass it as `--root` or `$OBS_REPO`.
- **PARSEC-UFO-stockholm**, compiled (see below). Pass it as `--parsec` or `$PARSEC_DIR`.

Generated data (`inputs/`, `runs/`, `results/`, `updrafts/`, `logs/`) is not tracked.

## Steps

1. **Inputs**: screened cloud hours (same screens as the susceptibility quicklooks),
   hourly spectrum on the common 10–1000 nm grid, composition, cloud-base T/P:

   ```
   export OBS_REPO=/path/to/obs-ccn-nd-susceptibility
   python build_parcel_inputs.py --sites ena hyytiala puijo     # or --all
   ```

   Composition is a per-regime placeholder (`comp_source = regime_placeholder`) until
   the CAMS / ACSM stage replaces it. T/P come from the built file where it has
   cloud-base T and P (ARM NDROP sites); elsewhere a placeholder until the
   thermodynamics stage (`thermo_source`).

2. **PARSEC runs**: every hour at 10 fixed updrafts (0.05–3 m/s), as a SLURM array.
   Submit from the repo root:

   ```
   mkdir -p logs
   N=$(python parsec_run.py --inputs inputs/ena_parcel_inputs.nc --chunk-hours 50 --print-ntasks)
   sbatch --array=0 parsec_array.sbatch ena            # one test task, then: seff <jobid>
   sbatch --array=1-$((N-1)) parsec_array.sbatch ena
   ```

   Output: `runs/<site>/chunk_*.csv`, one row per (hour, w): activated fraction for
   PARSEC's four criteria, Nd = AF × input number (cm-3 at start density), Smax,
   smallest activated dry radius, status. Runs are done in node-local scratch and
   their NetCDF output is deleted.

3. **Updraft weighting**: Nd(w) interpolated over each hour's measured w PDF
   (or a σ_w), so updraft data can change without rerunning PARSEC:

   ```
   python weight_updrafts.py --site ena --w-file updrafts/ena_w_cloudbase.nc
   python weight_updrafts.py --site ena --sigma-w 0.3          # sensitivity / no lidar yet
   ```

## Measured composition: what exists

```
export OBS_REPO=/path/to/obs-ccn-nd-susceptibility      # for ebas_psd_search.py and arm_ena_nd_pnsd.py
export ARM_USER=...; export ARM_TOKEN=...               # ARM Live, for the ARM sites
python composition_search.py                            # first run crawls the EBAS catalogue (minutes), then cached
```

For every site with an inputs file: ACSM/AMS, filter-pack ions, OC/EC, black carbon and CCN datasets
(EBAS stations within 30 km; ARM AOS datastreams), with the number of cloud hours inside each
dataset's period. Writes `results/composition_inventory.csv` and `..._summary.csv`.

## Measured composition: download

```
export ARM_USER=...; export ARM_TOKEN=...
python obs_composition_download.py --dry-run
nohup python -u obs_composition_download.py > logs/obs_comp_download.log 2>&1 &
```

ARM ACSM, PSAP/SP2 and CCN files for the cloud days (`data/obs_comp/arm/`), and the EBAS ACSM,
filter-ion, black-carbon and CCN files listed by `composition_search.py` (`data/obs_comp/ebas/`).
`data/obs_comp/manifest.csv` maps files to sites and categories.

## CAMS EAC4 composition (for hours without measured composition)

```
python cams_download.py --mode monthly --dry-run      # EAC4 monthly means: one request per region and year (fast)
python cams_download.py --dry-run                     # 3-hourly EAC4: one request per region and month, cloud days only
nohup python -u cams_download.py --mode monthly > logs/cams_monthly.log 2>&1 &    # the two modes use different
nohup python -u cams_download.py --mode daily   > logs/cams_daily.log 2>&1 &      # datasets, so both can run
```

Sites are grouped into five regional boxes (Europe, SGP + Bankhead, EPCAPE, ENA, LASIC), because ADS
reads whole global fields from tape whatever the area. Needs `~/.cdsapirc` for the Atmosphere Data
Store and the EAC4 (and EAC4 monthly) licences accepted. Aerosol species only (sulfate, organic
matter, black carbon, sea salt, dust); T and humidity come from ERA5. EAC4 currently ends in
December 2025; later days are listed, not requested.

## Composition per hour

```bash
python build_composition.py --dry-run      # tiers per site, nothing written
python build_composition.py                # writes massfrac, comp_source, m_* (ug m-3), kappa_bulk into inputs/
```

Hierarchy per hour: hourly ACSM (+ measured eBC, CAMS sea salt and dust) > daily EBAS filter ions
(nss sulfate, ammonium, 3.26 x Na scaled to its submicron share with CAMS, organics = CAMS OM/SO4 x
measured sulfate) > measured eBC + CAMS > CAMS only. `comp_source` records the tier, where BC came
from and which CAMS product filled the rest (`cams3h` > `camsmon` > `camsclim` for hours after EAC4).
Sulfate is split between H2SO4 / NH4HSO4 / (NH4)2SO4 from the NH4:SO4 molar ratio; ACSM nitrate and
chloride go into the (NH4)2SO4 column as NH4NO3 / NH4Cl mass. eBC = absorption / MAC (6.6 m2 g-1 at
637 nm, scaled 1/lambda). The placeholder fractions are kept as `massfrac_placeholder`. Rerun it when
more ARM or CAMS files arrive; it always rebuilds from the placeholder.

## Hygroscopicity from the CCN counters

```bash
python build_composition.py                   # kappa_bulk from the composition
python build_ccn_kappa.py                     # kappa_ccn per hour and setpoint, CCN closure -> results/<site>_ccn_kappa.nc
python build_composition.py --ccn-tier        # optional: CCN-constrained composition where only BC/CAMS exist
python build_ccn_kappa.py                     # closure again with the final composition
```

kappa_ccn comes from the critical dry diameter D_c at which the hourly spectrum holds as many
particles as the counter measured (kappa = 4A^3 / (27 D_c^3 ln^2 S), T = 298 K). The closure compares
the counter with CCN predicted from the spectrum and kappa_bulk, per setpoint and composition tier.
With `--ccn-tier`, hours that would get only BC + CAMS or CAMS composition and have a valid kappa_ccn
at the setpoint nearest `--ccn-ss` (default 0.3 %) get their (NH4)2SO4 / organics split re-chosen so
the bulk kappa matches; BC, dust and sea salt are unchanged; `comp_source` starts with `ccn`.

## Cloud-base temperature and pressure (ERA5)

```bash
python era5_download.py --dry-run                        # requests per ERA5 cell and month (cloud days)
nohup python -u era5_download.py > logs/era5.log 2>&1 &   # needs ~/.cdsapirc_cds (CDS url + key)
python met_download.py --dry-run                         # surface met: ARM MET, Cloudnet weather-station, FMI
nohup python -u met_download.py > logs/met_download.log 2>&1 &
python build_thermo.py                                   # T0/P0 for hours without measured cloud-base values
```

Surface T, RH, p are taken along a dry adiabat to the measured cloud base where cloud base is within
300 m of the surface LCL (coupled); in-cloud stations use their own T and p; ERA5 covers decoupled hours
and sites without surface met.

ERA5 T and p are interpolated to site altitude + cloud base height (Cloudnet, NDROP) or to the station
(in-cloud stations). Measured cloud-base values (NDROP built files, Zeppelin station) are kept; at the
NDROP sites the script reports ERA5 minus measured as a check. ERA5 is on the CDS, not the ADS, so its
key lives in a separate rc file; `~/.cdsapirc` stays pointed at the ADS for CAMS.

## Zeppelin: Bolin Centre cloud-residual dataset

```bash
wget -P data/bolin https://bolin.su.se/data/uploads/karlsson-2021-1.zip && unzip -o data/bolin/karlsson-2021-1.zip -d data/bolin
python zeppelin_bolin.py     # updrafts/zeppelin_w_cloudbase.nc, T0/P0 in inputs, results/zeppelin_activation.nc
```

Karlsson et al. (2021), doi:10.17043/zeppelin-cloud-aerosol-1. The station is in cloud, so its vertical
wind (5-7 min means; sigma_w is a lower bound) and temperature stand in for cloud-base values. Cloud
residuals / whole-air aerosol give the observed activation diameter D50 per hour, to compare with
PARSEC's activation radius.

## PARSEC settings

- adiabatic ascent, constant updraft, start RH 98 % at T0/P0, end 150 m above cloud base
- one internally mixed aerosol type; no coalescence, aqueous chemistry, SVOCs or compressed film
- concentrations written in **m-3** (PARSEC divides by air density in g m-3; the Python
  helper's "cm-3" docstring is wrong). Each run checks PARSEC's initial number against
  the input spectrum (`unit_mismatch` status if > 5 % off)
- measured grid bins split in two (`--subdivide 2`), ~130 bins per run, ~1 s CPU per run

## PARSEC build on Arrhenius

```
ml buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 OpenMPI/5.0.8 netCDF-Fortran/4.6.2
make FC=gfortran INCLUDE="$(nf-config --fflags)" LIBS="$(nf-config --flibs)"
```

Patch applied to `src/mo_parsec_output.f90` line 1529: a guard `IF (norg.ge.1)` around the
write of `xliq(isv,...)`, which is out of bounds when the SVOC scheme is off:

```
sed -i '1529s/^\(\s*\)IF (ANY(loc(292)==cv))/\1IF (norg.ge.1) THEN; IF (ANY(loc(292)==cv))/; 1529s/$/; END IF/' src/mo_parsec_output.f90
```

Lines 293 onwards have a related off-by-one (`isv+k` guarded by `norg.ge.k`) that only
matters with SVOCs on.

Second patch, `src/mo_parsec_kohler.f90` line 883: the dummy argument `ra_lims` of
`calculate_kohler_spectrum` is declared `(na)` but the array is `(na+1)` bin edges. When the
first activated bin of the initial CCN-spectrum diagnostic is the largest bin, the code reads
`ra_lims(na+1)` and the bounds check stops the run (seen in ~5 % of SGP hours):

```
sed -i '883s/ra_lims(na) /ra_lims(na+1) /' src/mo_parsec_kohler.f90
```

## Credit

PARSEC: P. Bowen, D. Partridge (University of Exeter), G.-J. Roelofs; Stockholm fork with
co-condensation and the unified organics framework. Check the PARSEC code-sharing policy
(attribution / co-authorship) before publishing results from these runs.
