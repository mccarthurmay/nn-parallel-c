#!/bin/sh

module load gcc
module load tau papi


export TAU_OPTIONS="-optCompInst -optTauSelectFile=select.tau -optVerbose"

rm -f ./*.o network_tau
CC=tau_cc.sh  
$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O2 -o network_tau_v1_O2 main.c network_tau_v1.c mnist_loader.c sigFuncs.c -lm
$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O3 -o network_tau_v1 main.c network_tau_v1.c mnist_loader.c sigFuncs.c -lm
$CC -Wall -Wextra -std=c11 -DTAU_ENABLED -O3 -o network_tau_v2 mainV2.c network_tau_v2.c mnist_loader.c sigFuncs.c -lm

echo ok