/*
Gradient check for backprop.

Backprop fails quietly: a sign error or an off-by-one still runs, it just
never learns. So rather than testing backprop against hand-computed numbers,
this compares every gradient it produces against a finite-difference estimate
of the same quantity, which only needs feedforward to be correct.

For each weight w, nudge it by +/- eps and measure how the cost actually moves:

    dC/dw  ~=  (C(w + eps) - C(w - eps)) / (2 * eps)

That numeric estimate and backprop's analytic nabla should agree to ~6 decimal
places in double precision.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "network.h"

#define EPS 1e-5
#define TOL 1e-6

static int failures = 0;

static void check(int cond, const char *what){
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) failures++;
}

/*
Quadratic cost for one example: C = 0.5 * sum((a - y)^2).
This is the cost whose derivative is (a - y), which is what cost_derivative
returns, so the two have to agree for the check to mean anything.
*/
static double cost(const Network *net, const double *x, unsigned char label,
                   double *out, double *scratch){
    feedforward(net, x, out, scratch);

    double c = 0.0;
    for (int k = 0; k < net->sizes[net->num_layers - 1]; k++){
        double y = (k == label) ? 1.0 : 0.0;
        c += (out[k] - y) * (out[k] - y);
    }
    return 0.5 * c;
}

// relative error, guarded so two near-zero gradients don't blow it up
static double rel_err(double a, double b){
    double denom = fabs(a) + fabs(b);
    if (denom < 1e-12) {
        return 0.0;
    }
    return fabs(a - b) / denom;
}

int main(void){
    srand(42);   // fixed seed: a failure here should be reproducible

    int sizes[3] = {3, 4, 2};
    int num_layers = 3;
    Network *net = network_init(sizes, num_layers);
    check(net != NULL, "network_init returned a network");
    if (net == NULL) return 1;

    Workspace *ws = workspace_init(net);
    check(ws != NULL, "workspace_init returned a workspace");
    if (ws == NULL) return 1;

    // one fixed example; x is float to match the Dataset layout backprop reads
    float x[3] = {0.35f, 0.9f, 0.2f};
    double xd[3];
    for (int i = 0; i < 3; i++) xd[i] = x[i];
    unsigned char label = 1;

    // gradient accumulators, same shape as the network's own weights/biases
    double **nabla_b = calloc(num_layers - 1, sizeof(double*));
    double **nabla_w = calloc(num_layers - 1, sizeof(double*));
    for (int l = 0; l < num_layers - 1; l++){
        nabla_b[l] = calloc(sizes[l + 1], sizeof(double));
        nabla_w[l] = calloc(sizes[l] * sizes[l + 1], sizeof(double));
    }

    backprop(net, x, label, nabla_b, nabla_w, ws);

    double out[8], scratch[16];
    double worst_w = 0.0, worst_b = 0.0;

    for (int l = 0; l < num_layers - 1; l++){
        // weights
        for (int i = 0; i < sizes[l] * sizes[l + 1]; i++){
            double saved = net->weights[l][i];

            net->weights[l][i] = saved + EPS;
            double c_plus = cost(net, xd, label, out, scratch);

            net->weights[l][i] = saved - EPS;
            double c_minus = cost(net, xd, label, out, scratch);

            net->weights[l][i] = saved;

            double numeric = (c_plus - c_minus) / (2.0 * EPS);
            double e = rel_err(numeric, nabla_w[l][i]);
            if (e > worst_w) worst_w = e;
        }

        // biases
        for (int i = 0; i < sizes[l + 1]; i++){
            double saved = net->biases[l][i];

            net->biases[l][i] = saved + EPS;
            double c_plus = cost(net, xd, label, out, scratch);

            net->biases[l][i] = saved - EPS;
            double c_minus = cost(net, xd, label, out, scratch);

            net->biases[l][i] = saved;

            double numeric = (c_plus - c_minus) / (2.0 * EPS);
            double e = rel_err(numeric, nabla_b[l][i]);
            if (e > worst_b) worst_b = e;
        }
    }

    printf("  worst relative error: weights %.3e, biases %.3e\n", worst_w, worst_b);
    check(worst_w < TOL, "weight gradients match finite differences");
    check(worst_b < TOL, "bias gradients match finite differences");

    /*
    backprop accumulates with += so a mini-batch can sum into one pair of
    arrays. Running the same example twice must therefore exactly double every
    gradient -- if it overwrote instead, the values would be unchanged.
    */
    double first = nabla_w[0][0];
    backprop(net, x, label, nabla_b, nabla_w, ws);
    check(fabs(nabla_w[0][0] - 2.0 * first) < 1e-12,
          "a second backprop accumulates rather than overwriting");

    /*
    A correct label the network is already confident about should push a much
    smaller gradient than a wrong one. This catches a one-hot built at the
    wrong index, which the finite-difference check cannot see (it would agree
    with a consistently wrong cost).
    */
    for (int l = 0; l < num_layers - 1; l++){
        for (int i = 0; i < sizes[l + 1]; i++) nabla_b[l][i] = 0.0;
        for (int i = 0; i < sizes[l] * sizes[l + 1]; i++) nabla_w[l][i] = 0.0;
    }
    feedforward(net, xd, out, scratch);
    unsigned char predicted = (out[0] > out[1]) ? 0 : 1;
    backprop(net, x, predicted, nabla_b, nabla_w, ws);
    double agree = fabs(nabla_b[num_layers - 2][predicted]);

    for (int l = 0; l < num_layers - 1; l++){
        for (int i = 0; i < sizes[l + 1]; i++) nabla_b[l][i] = 0.0;
        for (int i = 0; i < sizes[l] * sizes[l + 1]; i++) nabla_w[l][i] = 0.0;
    }
    backprop(net, x, (unsigned char)(1 - predicted), nabla_b, nabla_w, ws);
    double disagree = fabs(nabla_b[num_layers - 2][predicted]);

    printf("  output-delta magnitude: matching label %.4f, wrong label %.4f\n",
           agree, disagree);
    check(agree < disagree, "the wrong label produces the larger output gradient");

    for (int l = 0; l < num_layers - 1; l++){
        free(nabla_b[l]);
        free(nabla_w[l]);
    }
    free(nabla_b);
    free(nabla_w);
    workspace_destroy(ws);
    network_destroy(net);
    workspace_destroy(NULL);
    check(1, "workspace_destroy(NULL) did not crash");

    printf("\n%s\n", failures == 0 ? "all tests passed" : "TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
