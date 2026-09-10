#!/bin/sh
set -eu
module load gcc
module load tau papi

python3 mnist_loader.py

rm -f ./*.o network_v1 network_v2 network_tau_v1 network_tau_v1_O2 network_tau_v2

gcc -Wall -Wextra -std=c11 -O2 -o network_v1 main.c networkV1.c mnist_loader.c sigFuncs.c -lm
gcc -Wall -Wextra -std=c11 -O2 -o network_v2 main.c networkV2.c mnist_loader.c sigFuncs.c -lm
export TAU_OPTIONS="-optCompInst -optTauSelectFile=select.tau -optVerbose"


CC=tau_cc.sh
#$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O2 -o network_tau_v1_O2 main.c   networkV1.c mnist_loader.c sigFuncs.c -lm
$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O3 -o networkV1 main.c networkV1.c mnist_loader.c sigFuncs.c -lm
$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O3 -o networkV2 main.c networkV2.c mnist_loader.c sigFuncs.c -lm
$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O3 -o networkV3 main.c networkV3.c mnist_loader.c sigFuncs.c -lm
for b in network_tau_v1 network_tau_v1_O2 network_tau_v2; do
    printf '%s: %s TAU symbols\n' "$b" "$(nm "$b" | grep -c Tau_)"
done


echo ok
