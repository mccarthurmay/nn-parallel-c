#include <stdio.h>
#include <math.h>

#include "network.h"

int main(void) {
    double w[] = {1, 2, 3, 4};     /* [[1,2],[3,4]] */
    double a[] = {1, 1};
    double out[2];

    matmult(w, a, out, 2, 2);
    printf("matmult:  [%.1f, %.1f]   expected [3.0, 7.0]\n", out[0], out[1]);

    int sizes[] = {2, 2};
    double b[] = {0.5, -0.5};
    double *wp = w, *bp = b;
    Network net = {2, sizes, 2, &bp, &wp};

    double scratch[4];
    feedforward(&net, a, out, scratch);
    printf("ff:       [%.15f, %.15f]\n", out[0], out[1]);
    printf("expected: [%.15f, %.15f]\n", 1/(1+exp(-3.5)), 1/(1+exp(-6.5)));

    return 0;
}