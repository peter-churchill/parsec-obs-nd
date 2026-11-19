#!/bin/bash -l
# PARSEC-UFO parcel runs for one site as a SLURM array (one chunk of cloud hours per task).
#
# Submit from the repo root:
#   N=$(python parsec_run.py --inputs inputs/<site>_parcel_inputs.nc --chunk-hours 50 --print-ntasks)
#   sbatch --array=0-$((N-1)) parsec_array.sbatch <site>
#
# Test first with one task and check it with `seff <jobid>`:
#   sbatch --array=0 parsec_array.sbatch ena
#
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 01:00:00
#SBATCH -J parsec
#SBATCH -o logs/parsec_%x_%A_%a.out

set -eo pipefail     # no -u: conda activate scripts reference unset variables
SITE=${1:?usage: sbatch --array=... parsec_array.sbatch <site>}
CHUNK=${CHUNK:-50}

# --- edit these once (or export them before sbatch) ---
REPO=${REPO:-/nobackup/proj/disk/luftlab/personal/petchu/parsec-obs-nd}
PARSEC_DIR=${PARSEC_DIR:-/nobackup/proj/disk/luftlab/personal/petchu/PARSEC-UFO-stockholm}
CONDA_ENV=${CONDA_ENV:-analysis}

# Library path for the PARSEC binary only (it was built against these modules); the Python
# side keeps its own conda libraries, so the two never mix.
module purge
ml buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 OpenMPI/5.0.8 netCDF-Fortran/4.6.2
export PARSEC_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
module purge

# login shell (-l above) sets up conda as in your terminal
conda activate "$CONDA_ENV"

cd "$REPO"
python -u parsec_run.py \
    --inputs "inputs/${SITE}_parcel_inputs.nc" \
    --parsec "$PARSEC_DIR" \
    --chunk-hours "$CHUNK" \
    --task "$SLURM_ARRAY_TASK_ID" \
    --outdir runs
