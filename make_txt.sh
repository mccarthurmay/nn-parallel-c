#!/bin/bash
set -euo pipefail

for tau_dir in tau_*/; do
    [ -d "$tau_dir" ] || continue
    for metric_dir in "$tau_dir"*/; do
        [ -f "${metric_dir}profile.0.0.0" ] || continue
        out="${metric_dir}profile.txt"
        ( cd "$metric_dir" && pprof -a ) > "$out"
        echo "wrote $out"
    done
done