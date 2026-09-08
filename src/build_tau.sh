#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# as/ld are not reliably on PATH in a fresh login shell
export PATH="/usr/bin:$PATH"

module load tau papi
export TAU_OPTIONS="-optCompInst -optVerbose"

echo "--- toolchain ---"
which gcc as ld || { echo "as/ld still missing" >&2; exit 1; }

# stale objects from a half-finished build will silently get linked
rm -f ./*.o network_tau

tau_cc.sh -B/usr/bin -DTAU_ENABLED -O2 -std=c11 \
    -o network_tau main.c network.c mnist_loader.c sigFuncs.c -lm 2>&1 | tee tau_build.log

echo "--- TAU symbols (must be > 0) ---"
nm network_tau | grep -c Tau_
