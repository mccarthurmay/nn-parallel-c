/*
Equivalence check for batch_backprop.

batch_backprop reassociates exactly the same arithmetic backprop already does:
the same per-example gradients, summed in a different order. So it is not
checked against finite differences here -- test_backprop.c already established
that backprop itself is correct. Instead this pins batch_backprop to backprop,
which is a much tighter constraint and catches transposes, index swaps, and
missing accumulations that a finite-difference check would be too loose to see.

  Check 1: m = 1 must reproduce a single backprop call almost exactly. With one
           example there is no reordering at all, so the only differences are
           sigmoid_prime(z) vs the algebraically identical a*(1-a).

  Check 2: m = 10 must equal ten backprop calls accumulated into one pair of
           arrays. Here the summation order genuinely differs, so the tolerance
           is looser -- but it is still the strongest statement available: the
           batched path computes the same mini-batch gradient.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "network.h"

#define TOL_ONE   1e-12   // m = 1: identical arithmetic, only sigmoid' differs
#define TOL_BATCH 1e-10   // m > 1: summation order differs

#define NEX 10            // examples in the synthetic dataset
#define DIM 3             // pixels per example

static int failures = 0;

static void check(int cond, const char *what){
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) failures++;
}

static double rel_err(double a, double b){
    double denom = fabs(a) + fabs(b);
    if (denom < 1e-12) return 0.0;
    return fabs(a - b) / denom;
}

// allocate zeroed gradient arrays shaped like the network's weights/biases
static void grads_alloc(const int *sizes, int num_layers,
                        double ***nb, double ***nw){
    *nb = calloc(num_layers - 1, sizeof(double*));
    *nw = calloc(num_layers - 1, sizeof(double*));
    for (int l = 0; l < num_layers - 1; l++){
        (*nb)[l] = calloc(sizes[l + 1], sizeof(double));
        (*nw)[l] = calloc(sizes[l] * sizes[l + 1], sizeof(double));
    }
}

static void grads_free(double **nb, double **nw, int num_layers){
    for (int l = 0; l < num_layers - 1; l++){ free(nb[l]); free(nw[l]); }
    free(nb); free(nw);
}

// worst relative error between two gradient sets, over every weight and bias
static double grads_diff(double **nb_a, double **nw_a,
                         double **nb_b, double **nw_b,
                         const int *sizes, int num_layers){
    double worst = 0.0;
    for (int l = 0; l < num_layers - 1; l++){
        for (int i = 0; i < sizes[l] * sizes[l + 1]; i++){
            double e = rel_err(nw_a[l][i], nw_b[l][i]);
            if (e > worst) worst = e;
        }
        for (int i = 0; i < sizes[l + 1]; i++){
            double e = rel_err(nb_a[l][i], nb_b[l][i]);
            if (e > worst) worst = e;
        }
    }
    return worst;
}


int main(void){
    srand(42);

    int sizes[3] = {DIM, 4, 2};
    int num_layers = 3;

    Network *net = network_init(sizes, num_layers);
    check(net != NULL, "network_init returned a network");
    if (net == NULL) return 1;

    Workspace *ws = workspace_init(net);
    BatchWorkspace *bws = batch_workspace_init(net, NEX);
    check(ws != NULL, "workspace_init returned a workspace");
    check(bws != NULL, "batch_workspace_init returned a workspace");
    if (ws == NULL || bws == NULL) return 1;

    /*
    A synthetic Dataset. Dataset is a plain struct of caller-owned pointers, so
    it can be built on the stack -- no .bin file and no loader needed.
    */
    float pixels[NEX * DIM];
    unsigned char labels[NEX];
    for (int i = 0; i < NEX; i++){
        for (int j = 0; j < DIM; j++){
            pixels[i * DIM + j] = (float)((i * DIM + j) % 7) / 7.0f;
        }
        labels[i] = (unsigned char)(i % 2);
    }
    Dataset data = { NEX, DIM, pixels, labels };

    int idx[NEX];
    for (int i = 0; i < NEX; i++) idx[i] = i;

    double **nb_old, **nw_old, **nb_new, **nw_new;

    /* ---- Check 1: m = 1 matches a single backprop ---- */
    grads_alloc(sizes, num_layers, &nb_old, &nw_old);
    grads_alloc(sizes, num_layers, &nb_new, &nw_new);

    backprop(net, pixels, labels[0], nb_old, nw_old, ws);
    batch_backprop(net, &data, idx, 1, nb_new, nw_new, bws);

    double worst1 = grads_diff(nb_old, nw_old, nb_new, nw_new, sizes, num_layers);
    printf("  m=1  worst relative error: %.3e\n", worst1);
    check(worst1 < TOL_ONE, "batch_backprop with m=1 matches backprop");

    grads_free(nb_old, nw_old, num_layers);
    grads_free(nb_new, nw_new, num_layers);

    /* ---- Check 2: m = NEX matches NEX accumulated backprop calls ---- */
    grads_alloc(sizes, num_layers, &nb_old, &nw_old);
    grads_alloc(sizes, num_layers, &nb_new, &nw_new);

    for (int i = 0; i < NEX; i++){
        backprop(net, pixels + (size_t)i * DIM, labels[i], nb_old, nw_old, ws);
    }
    batch_backprop(net, &data, idx, NEX, nb_new, nw_new, bws);

    double worst2 = grads_diff(nb_old, nw_old, nb_new, nw_new, sizes, num_layers);
    printf("  m=%d worst relative error: %.3e\n", NEX, worst2);
    check(worst2 < TOL_BATCH, "batch_backprop matches summed backprop calls");
    /*
    The gradients must actually be non-trivial, otherwise the two checks above
    would pass on two identically-zero arrays.
    */
    double mag = 0.0;
    for (int i = 0; i < sizes[0] * sizes[1]; i++) mag += fabs(nw_new[0][i]);
    printf("  total |nabla_w[0]| = %.4f\n", mag);
    check(mag > 1e-6, "gradients are non-zero");

    /*
    batch_backprop must accumulate with += like backprop does, so that
    update_mini_batch can zero once and sum across the batch.
    */
    double before = nw_new[0][0];
    batch_backprop(net, &data, idx, NEX, nb_new, nw_new, bws);
    check(fabs(nw_new[0][0] - 2.0 * before) < 1e-9,
          "a second batch_backprop accumulates rather than overwriting");

    grads_free(nb_old, nw_old, num_layers);
    grads_free(nb_new, nw_new, num_layers);

    batch_workspace_destroy(bws);
    workspace_destroy(ws);
    network_destroy(net);
    batch_workspace_destroy(NULL);
    check(1, "batch_workspace_destroy(NULL) did not crash");

    printf("\n%s\n", failures == 0 ? "all tests passed" : "TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
