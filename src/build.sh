#!/bin/sh

gcc -Wall -Wextra -std=c11 -O2 -o network_v1 main.c network_tau_v1.c mnist_loader.c sigFuncs.c -lm
gcc -Wall -Wextra -std=c11 -O2 -o network_v2 main.c network_tau_v2.c mnist_loader.c sigFuncs.c -lm

echo ok
