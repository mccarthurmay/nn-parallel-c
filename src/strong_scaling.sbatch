#!/bin/bash
#SBATCH --job-name=nn-strong
#SBATCH --output=strong_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=Cpu40
#SBATCH --exclusive


set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

EPOCHS=3
ETA=3.0
REPS=3
CORE_COUNTS=(1 2 4 8 16 24 32)
BATCH=25

HIDDEN_SPECS=("200")

AVAIL=$(nproc)
echo "cores on this node: $AVAIL"

shopt -s nullglob
BINS=(network_v*)
shopt -u nullglob

if [ ${#BINS[@]} -eq 0 ]; then
    echo "no network_v* binaries found in $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

# run <binary> <csv> <hidden> <batch> <cores>
run(){
    bin=$1
    csv=$2
    hidden=$3
    batch=$4
    cores=$5
    log=$(mktemp)

    start=$(date +%s.%N)
    if ! srun --ntasks=1 --cpus-per-task="$cores" --cpu-bind=cores \
              --export=ALL,OMP_NUM_THREADS="$cores",OMP_PROC_BIND=close,OMP_PLACES=cores \
              "./$bin" "$EPOCHS" "$batch" "$ETA" $hidden > "$log" 2>&1; then
        echo "FAILED: bin=$bin hidden=$hidden batch=$batch cores=$cores" >&2
        sed 's/^/    /' "$log" >&2
        rm -f "$log"
        return 0       
    fi
    end=$(date +%s.%N)

    # main.c prints "Epoch N: <correct> / <total>" per epoch; take the last
    last=$(grep '^Epoch' "$log" | tail -1)
    correct=$(echo "$last" | awk '{print $3}')
    total=$(echo "$last" | awk '{print $5}')
    rm -f "$log"

    if [ -z "$correct" ] || [ -z "$total" ]; then
        echo "no epoch line parsed: bin=$bin hidden=$hidden cores=$cores" >&2
        return 0
    fi

    awk -v h="$hidden" -v b="$batch" -v e="$ETA" -v ep="$EPOCHS" -v co="$cores" \
        -v s="$start" -v f="$end" -v c="$correct" -v t="$total" -v n="$bin"\
        'BEGIN { w = f - s;
                 printf "%s,%s,%d,%s,%d,%d,%.3f,%.3f,%d,%d,%.4f\n",
                        n, h, b, e, ep, co, w, w/ep, c, t, c/t }' >> "$csv"

    printf '  hidden=%-6s cores=%-3s  %ss  %s/%s\n' \
           "$hidden" "$cores" \
           "$(awk -v s="$start" -v f="$end" 'BEGIN{printf "%.1f", f-s}')" \
           "$correct" "$total"
}

CSV="$SLURM_SUBMIT_DIR/strong_${SLURM_JOB_ID:-local}.csv"
echo "network,hidden,batch,eta,epochs,cores,wall_s,s_per_epoch,correct,total,accuracy" > "$CSV"

for bin in "${BINS[@]}"; do
    [ -x "$bin" ] || { echo "skipping $bin (not executable)" >&2; continue; }

    echo "########################################"
    echo "### $bin"
    echo "########################################"

    for hidden in "${HIDDEN_SPECS[@]}"; do
        echo "=== core sweep (hidden=$hidden, batch=$BATCH) ==="
        for rep in $(seq 1 "$REPS"); do
            for c in "${CORE_COUNTS[@]}"; do
                [ "$c" -le "$AVAIL" ] || continue
                run "$bin" "$CSV" "$hidden" "$BATCH" "$c"
            done
        done
    done

    echo
done

echo "wrote $CSV"
column -s, -t < "$CSV"

