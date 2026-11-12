#!/bin/bash
# Pure-Si periodic box: eigenvalues nearest SIGMA with silicon_unstrained.py
# (ARPACK shift-invert, Pardiso LDL^H inner solver).
#
# Usage:   sbatch -w x8 job_unstrained.sh <ncells> [sigma_eV] [k]
# Example: sbatch -w x8 job_unstrained.sh 10
#          sbatch -w x9 job_unstrained.sh 12 1.173 60
#
# Note: ARPACK can drop members of degenerate (12-fold) shells.
# Request ~2x the states you need and trust only the lowest complete levels;
# for guaranteed-complete multiplets use silicon_pure_box.py (LOBPCG).

#SBATCH -J si-unstr
#SBATCH -o unstr.%j.out
#SBATCH -e unstr.%j.err
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40        # whole x8/x9 node (20 cores x 2 hyperthreads)
# SBATCH --mem=58G
#SBATCH -t 24:00:00

N=${1:?usage: sbatch -w <node> job_unstrained.sh <ncells> [sigma_eV] [k]}
SIGMA=${2:-1.173}
K=${3:-40}

# One Python process; MKL Pardiso and BLAS use the threads (physical cores).
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

source /u0/anirban/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Python   : $(which python3)"
echo "Threads  : $OMP_NUM_THREADS"
echo "Run      : ncells=$N  sigma=$SIGMA eV  k=$K"
echo "=========================================="
python3 -c "import pydiso; print('pydiso OK (native Hermitian Pardiso)')" \
    || echo "WARNING: pydiso missing -> falls back to pypardiso real-2N (≈3.5x memory)"

# -u: unbuffered output, so unstr.<jobid>.out updates live
python3 -u silicon_unstrained.py --ncells "$N" --sigma "$SIGMA" --k "$K"
