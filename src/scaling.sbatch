#!/bin/bash
#SBATCH --job-name=nn-scaling
#SBATCH --output=scaling.out
#SBATCH --time=02:00:00
#SBATCH --partition=Mem500
#SBATCH --exclusive

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"


CSV="$SLURM_SUBMIT_DIR/scaling.csv"
echo "hidden,batch,eta,epochs,threads,wall_s,s_per_epoch,correct,total,accuracy" > "$CSV"

EPOCHS=3          
ETA=3.0
REPS=3           
THREADS="${OMP_NUM_THREADS:-1}"


# run <hidden> <batch>
run(){
    hidden=$1
    batch=$2
    log=$(mktemp)

    start=$(date +%s.%N)
    if ! ./network "$EPOCHS" "$batch" "$ETA" "$hidden" > "$log" 2>&1; then
        echo "FAILED: hidden=$hidden batch=$batch" >&2
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
        echo "no epoch line parsed: hidden=$hidden batch=$batch" >&2
        return 0
    fi

    awk -v h="$hidden" -v b="$batch" -v e="$ETA" -v ep="$EPOCHS" -v th="$THREADS" \
        -v s="$start" -v f="$end" -v c="$correct" -v t="$total" \
        'BEGIN { w = f - s;
                 printf "%d,%d,%s,%d,%d,%.3f,%.3f,%d,%d,%.4f\n",
                        h, b, e, ep, th, w, w/ep, c, t, c/t }' >> "$CSV"

    printf 'hidden=%-4s batch=%-3s  %ss  %s/%s\n' \
           "$hidden" "$batch" \
           "$(awk -v s="$start" -v f="$end" 'BEGIN{printf "%.1f", f-s}')" \
           "$correct" "$total"
}

echo "=== hidden-layer sweep (batch fixed at 10) ==="
for rep in $(seq 1 "$REPS"); do
    for hidden in 10 30 50 100 200; do
        run "$hidden" 10
    done
done

echo "=== mini-batch sweep (hidden fixed at 30) ==="
for rep in $(seq 1 "$REPS"); do
    for batch in 1 5 10 25 50 100; do
        run 30 "$batch"
    done
done

echo
echo "wrote $CSV"
column -s, -t < "$CSV"
