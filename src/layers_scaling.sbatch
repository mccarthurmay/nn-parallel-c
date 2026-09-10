#!/bin/bash
#SBATCH --job-name=nn-layers-scaling
#SBATCH --output=layers_scaling_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=Mem500
#SBATCH --exclusive

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

EPOCHS=3
ETA=3.0
BATCH=10
REPS=3
THREADS="${OMP_NUM_THREADS:-1}"

shopt -s nullglob
BINS=(network_v*)
shopt -u nullglob

if [ ${#BINS[@]} -eq 0 ]; then
    echo "no network_v* binaries found in $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

# run <binary> <csv> <depth> <width>
run(){
    bin=$1
    csv=$2
    depth=$3
    width=$4
    log=$(mktemp)

    # depth copies of width -- main.c adds one hidden layer per trailing arg
    hidden=()
    for ((i=0; i<depth; i++)); do hidden+=("$width"); done

    start=$(date +%s.%N)
    if ! "./$bin" "$EPOCHS" "$BATCH" "$ETA" "${hidden[@]}" > "$log" 2>&1; then
        echo "FAILED: bin=$bin depth=$depth width=$width" >&2
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
        echo "no epoch line parsed: bin=$bin depth=$depth width=$width" >&2
        return 0
    fi

    awk -v d="$depth" -v w="$width" -v b="$BATCH" -v e="$ETA" -v ep="$EPOCHS" -v th="$THREADS" \
        -v s="$start" -v f="$end" -v c="$correct" -v t="$total" -v n="$bin" \
        'BEGIN { wall = f - s;
                 printf "%s,%d,%d,%d,%s,%d,%d,%.3f,%.3f,%d,%d,%.4f\n",
                        n, d, w, b, e, ep, th, wall, wall/ep, c, t, c/t }' >> "$csv"

    printf '  depth=%-3s width=%-4s  %ss  %s/%s\n' \
           "$depth" "$width" \
           "$(awk -v s="$start" -v f="$end" 'BEGIN{printf "%.1f", f-s}')" \
           "$correct" "$total"
}

CSV="$SLURM_SUBMIT_DIR/layers_scaling_${SLURM_JOB_ID:-local}.csv"
echo "network,depth,width,batch,eta,epochs,threads,wall_s,s_per_epoch,correct,total,accuracy" > "$CSV"

for bin in "${BINS[@]}"; do
    [ -x "$bin" ] || { echo "skipping $bin (not executable)" >&2; continue; }

    echo "########################################"
    echo "### $bin"
    echo "########################################"

    echo "=== depth sweep (width fixed at 32) ==="
    for rep in $(seq 1 "$REPS"); do
        for depth in 1 2 4 8 16 32; do
            run "$bin" "$CSV" "$depth" 32
        done
    done

    echo "=== width sweep (depth fixed at 2) ==="
    for rep in $(seq 1 "$REPS"); do
        for width in 16 32 64 128 256 512; do
            run "$bin" "$CSV" 2 "$width"
        done
    done

    echo "=== depth x width grid ==="
    for rep in $(seq 1 "$REPS"); do
        for depth in 1 2 4 8 16; do
            for width in 16 32 64 128; do
                run "$bin" "$CSV" "$depth" "$width"
            done
        done
    done

    echo
    echo "wrote $CSV"
    column -s, -t < "$CSV"
    echo
done

: <<'EOF'
Same idea as scaling.sbatch, but the knob is network DEPTH.

main.c takes <epochs> <batch> <eta> <hidden1> [hidden2 ...] and stacks one
hidden layer per trailing argument, so passing "32 32 32 32" builds a network
with four hidden layers of 32 neurons each. run() just repeats <width> <depth>
times to build that argument list.

Both depth and width go up in powers of two, so a log-log plot of time is a
straight line and the slope is the scaling exponent.

depth scaling
depth:  1, 2, 4, 8, 16, 32
width:  32
runs:   18

width scaling
depth:  2
width:  16, 32, 64, 128, 256, 512
runs:   18

depth x width grid
depth:  1, 2, 4, 8, 16
width:  16, 32, 64, 128
runs:   60

batch size: 10
epochs trained: 3
learning rate: 3

All of the above runs once per network_v* binary found in the submit dir.

Note: past ~8 hidden layers, plain sigmoid with no normalization will have
vanishing gradients and the accuracy column will flatten out near 10%. The
timing data is still good -- just don't read the deep end as a real accuracy
trend.
EOF
