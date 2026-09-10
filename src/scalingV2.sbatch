#!/bin/bash
#SBATCH --job-name=nn-scaling
#SBATCH --output=scaling_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=Mem500
#SBATCH --exclusive

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

EPOCHS=3
ETA=3.0
REPS=3
THREADS="${OMP_NUM_THREADS:-1}"
HIDDEN_SPECS=("30" "100" "200" "30 30" "100 30" "50 50 50")
HIDDEN_SWEEP_BATCH=10
BATCH_SIZES=(1 5 10 25 50 100)
BATCH_SWEEP_HIDDEN=30



shopt -s nullglob
BINS=(network_v*)
shopt -u nullglob

if [ ${#BINS[@]} -eq 0 ]; then
    echo "no network_v* binaries found in $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

# run <binary> <csv> <hidden> <batch>
run(){
    bin=$1
    csv=$2
    hidden=$3
    batch=$4
    log=$(mktemp)

    start=$(date +%s.%N)
    if ! "./$bin" "$EPOCHS" "$batch" "$ETA" $hidden > "$log" 2>&1; then
        echo "FAILED: bin=$bin hidden=$hidden batch=$batch" >&2
        sed 's/^/    /' "$log" >&2
        rm -f "$log"
        return 0        # keep the sweep going rather than killing the job
    fi
    end=$(date +%s.%N)

    # main.c prints "Epoch N: <correct> / <total>" per epoch; take the last
    last=$(grep '^Epoch' "$log" | tail -1)
    correct=$(echo "$last" | awk '{print $3}')
    total=$(echo "$last" | awk '{print $5}')
    rm -f "$log"

    if [ -z "$correct" ] || [ -z "$total" ]; then
        echo "no epoch line parsed: bin=$bin hidden=$hidden batch=$batch" >&2
        return 0
    fi

    awk -v h="$hidden" -v b="$batch" -v e="$ETA" -v ep="$EPOCHS" -v th="$THREADS" \
        -v s="$start" -v f="$end" -v c="$correct" -v t="$total" -v n="$bin"\
        'BEGIN { w = f - s;
                 printf "%s,%s,%d,%s,%d,%d,%.3f,%.3f,%d,%d,%.4f\n",
                        n, h, b, e, ep, th, w, w/ep, c, t, c/t }' >> "$csv"

    printf '  hidden=%-12s batch=%-3s  %ss  %s/%s\n' \
           "$hidden" "$batch" \
           "$(awk -v s="$start" -v f="$end" 'BEGIN{printf "%.1f", f-s}')" \
           "$correct" "$total"
}

CSV="$SLURM_SUBMIT_DIR/scaling_${SLURM_JOB_ID:-local}.csv"
echo "network,hidden,batch,eta,epochs,threads,wall_s,s_per_epoch,correct,total,accuracy" > "$CSV"

for bin in "${BINS[@]}"; do
    [ -x "$bin" ] || { echo "skipping $bin (not executable)" >&2; continue; }


    echo "########################################"
    echo "### $bin"
    echo "########################################"

    echo "=== hidden-layer sweep (batch fixed at 10) ==="
    for rep in $(seq 1 "$REPS"); do
        for hidden in "${HIDDEN_SPECS[@]}"; do
            run "$bin" "$CSV" "$hidden" "$HIDDEN_SWEEP_BATCH"
        done
    done

    echo "=== mini-batch sweep (hidden fixed at 30) ==="
    for rep in $(seq 1 "$REPS"); do
        for batch in "${BATCH_SIZES[@]}"; do
            run "$bin" "$CSV" "$BATCH_SWEEP_HIDDEN" "$batch"
        done
    done

    echo
    echo "wrote $CSV"
    column -s, -t < "$CSV"
    echo
done

: <<'EOF'
I think this works like a comment, just using this space to like put into words what this is testing so i remember.
Hidden-layer scaling:
hidden layer: 10,30,50,100,200
batch size: 10
epochs trained: 3
learning rate: 3
runs: 15

mini-batch scaling
hidden layer: 30
batch size: 1, 5, 10, 25, 50, 100
epochs trained: 3
learning rate: 3
runs: 18

All of the above runs once per network_v* binary found in the submit dir.
EOF
