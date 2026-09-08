#!/bin/bash
#SBATCH --job-name=nn-tau
#SBATCH --output=tau.out
#SBATCH --time=02:00:00
#SBATCH --partition=Mem500
#SBATCH --exclusive



set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load tau papi


# Confirm counters exist
papi_avail | grep -E "PAPI_TOT_INS|PAPI_L1_DCM" || true

export TAU_METRICS=TIME:PAPI_TOT_INS:PAPI_L1_DCM
export TAU_PROFILE=1
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=4

EPOCHS=3
BATCH=10
ETA=3.0
HIDDEN=30

OUT="$SLURM_SUBMIT_DIR/tau_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

cd src
tau_cc.sh -DTAU_ENABLED -O2 -std=c11 \
    -o network_tau main.c network.c mnist_loader.c sigFuncs.c -lm

cd "$OUT"
"$SLURM_SUBMIT_DIR/src/network_tau" "$EPOCHS" "$BATCH" "$ETA" "$HIDDEN"

for d in MULTI__*; do
    echo "=============== $d ==============="
    ( cd "$d" && pprof -a) | tee "../&(basename "$d").txt"
done

echo "profiles in $OUT"



