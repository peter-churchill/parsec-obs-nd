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
