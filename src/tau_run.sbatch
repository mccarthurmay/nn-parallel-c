#!/bin/bash
#SBATCH --job-name=nn-tau
#SBATCH --output=tau_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=Mem500
#SBATCH --exclusive



set -euo pipefail

module load gcc
module load tau papi


# Confirm counters exist
papi_avail | grep -E "PAPI_TOT_INS|PAPI_L1_DCM" || true


export TAU_OPTIONS="-optCompInst -optVerbose"
export TAU_METRICS=TIME:PAPI_TOT_INS:PAPI_L1_DCM
export TAU_PROFILE=1
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=4

EPOCHS=3
BATCH=10
ETA=3.0
HIDDEN=30

BIN="$REPO/src/network_tau"
[ -x "$BIN" ] || { echo "no network_tau -- run: bash src/build_tau.sh" >&2; exit 1; }
nm "$BIN" | grep -q Tau_ || { echo "network_tau is not instrumented -- rebuild" >&2; exit 1; }

OUT="$REPO/tau_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
cd "$OUT"

"$BIN" "$EPOCHS" "$BATCH" "$ETA" "$HIDDEN"

for d in MULTI__*; do
    echo "=============== $d ==============="
    ( cd "$d" && pprof -a ) | tee "../$(basename "$d").txt"
done

echo "profiles in $OUT"

