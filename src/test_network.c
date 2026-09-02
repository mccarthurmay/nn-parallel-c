#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"

static int failures = 0;

static void check(int cond, const char *what){
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) failures++;
}

int main(void){
    srand(time(NULL));

    // small network: structure and defensive copy of sizes
    int sizes[3] = {2, 3, 1};
    Network *net = network_init(sizes, 3);
    check(net != NULL, "network_init returned a network");
    if (net == NULL) return 1;

    check(net->num_layers == 3, "num_layers is 3");
    check(net->sizes[0] == 2 && net->sizes[1] == 3 && net->sizes[2] == 1,
          "sizes copied correctly");

    sizes[0] = 99;
    check(net->sizes[0] == 2, "sizes is a copy, not the caller's array");

    // every weight and bias is a usable float, and they are not all the same
    int finite = 1, distinct = 0;
    for (int i=0; i<net->num_layers-1; i++){
        for (int j=0; j<net->sizes[i]*net->sizes[i+1]; j++){
            if (!isfinite(net->weights[i][j])) finite = 0;
            if (net->weights[i][j] != net->weights[0][0]) distinct = 1;
        }
        for (int j=0; j<net->sizes[i+1]; j++){
            if (!isfinite(net->biases[i][j])) finite = 0;
        }
    }
    check(finite, "all weights and biases are finite");
    check(distinct, "weights are not all identical");

    network_destroy(net);

    // large network: confirm randn is actually N(0, 1)
    int big[2] = {500, 500};
    Network *bignet = network_init(big, 2);
    check(bignet != NULL, "large network allocated");
    if (bignet == NULL) return 1;

    int n = 500 * 500;
    double sum = 0.0;
    for (int j=0; j<n; j++) sum += bignet->weights[0][j];
    double mean = sum / n;

    double sq = 0.0;
    for (int j=0; j<n; j++){
        double d = bignet->weights[0][j] - mean;
        sq += d * d;
    }
    double stddev = sqrt(sq / (n - 1));

    printf("  sample mean = %f, stddev = %f\n", mean, stddev);
    check(fabs(mean) < 0.1, "weight mean is near 0");
    check(fabs(stddev - 1.0) < 0.1, "weight stddev is near 1");

    network_destroy(bignet);

    // destroy tolerates NULL
    network_destroy(NULL);
    check(1, "network_destroy(NULL) did not crash");

    printf("\n%s\n", failures == 0 ? "all tests passed" : "TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
