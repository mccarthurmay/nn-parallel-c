#!/bin/bash
#SBATCH --job-name=nn-tau
#SBATCH --output=tau_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=Mem500
#SBATCH --exclusive

set -euo pipefail

# this script lives in src/, so the repo root may be one level up
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
if [ ! -d "$REPO/src" ] && [ -d "$REPO/../src" ]; then
    REPO=$(cd "$REPO/.." && pwd)
fi
[ -d "$REPO/src" ] || { echo "cannot find src/ from $REPO" >&2; exit 1; }

# needed at RUN time: libtau, libpapi, and libstdc++/libgcc_s from the spack gcc
module load gcc
module load tau papi

papi_avail | grep -E "PAPI_TOT_INS|PAPI_L1_DCM|PAPI_L2_DCM|PAPI_L3_TCM" || true

export TAU_METRICS=TIME:PAPI_L1_DCM:PAPI_L2_DCM:PAPI_L3_TCM:PAPI_TOT_INS
export TAU_PROFILE=1
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=4

EPOCHS=3
BATCH=10
ETA=3.0
HIDDEN_SPECS=("30")

# collect only executables; the glob would otherwise match .c and .o files too
BINS=()
for f in "$REPO"/src/network_tau_v3; do
    [ -f "$f" ] && [ -x "$f" ] && BINS+=("$f")
done
[ ${#BINS[@]} -gt 0 ] || {
    echo "no network_tau_v* binaries in $REPO/src -- run: bash src/build.sh" >&2
    exit 1
}

for BIN in "${BINS[@]}"; do
    [ -x "$BIN" ] || { echo "skipping $BIN (not executable)" >&2; continue; }

    NAME=$(basename "$BIN")
    for SPEC in "${HIDDEN_SPECS[@]}"; do
        read -ra HIDDEN <<< "$SPEC"

        SHAPE=$(IFS=-; echo "${HIDDEN[*]}")
        OUT="$REPO/tau_${SLURM_JOB_ID:-local}_${NAME}_${SHAPE}"
        mkdir -p "$OUT"
        cd "$OUT"

        echo "=== running $NAME hidden=[${HIDDEN[*]}] ==="
        "$BIN" "$EPOCHS" "$BATCH" "$ETA" "${HIDDEN[@]}"

        for d in MULTI__*; do
            [ -d "$d" ] || continue
            echo "--- $NAME $SHAPE: $d ---"
            ( cd "$d" && pprof -a ) | tee "../$(basename "$d").txt"
        done

        echo "profiles in $OUT"
        cd "$REPO"
    done
done
