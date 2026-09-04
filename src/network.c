/*
A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation.
*/

// Libraries
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "network.h"
#include "mnist_loader.h"
#include "sigFuncs.h"
#define PI 3.14159265358979323846

/*
Standard normal sample (mean 0, variance 1) via Box-Muller, matching
np.random.randn in the reference implementation. Callers are responsible
for seeding with srand() once at program start.
*/
static double randn(void){
    // shift off the endpoints so u1 is in (0,1) and logf never sees 0
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// Network constructor
Network *network_init(const int *sizes, int num_layers){
    Network *net = malloc(sizeof(Network));
    if (net == NULL) {
        return NULL;
    }
    // zeroed up front so network_destroy can clean up a partly built network
    memset(net, 0, sizeof(Network));

    net->num_layers = num_layers;
    net->sizes = malloc(num_layers * sizeof(int));
    if (net->sizes == NULL) {
        network_destroy(net);
        return NULL;
    }
    memcpy(net->sizes, sizes, num_layers * sizeof(int));
    

    net->max_size = 0;
    for (int i = 0; i < num_layers; i++){
        if (sizes[i] > net->max_size){
            net->max_size = sizes[i];
        }
    }
    // input layer doesnt have weights or biases, only output (receiving) layers
    net->weights = calloc(num_layers - 1, sizeof(double*));
    net->biases = calloc(num_layers - 1, sizeof(double*));
    if (net->weights == NULL || net->biases == NULL) {
        network_destroy(net);
        return NULL;
    }

    for (int i=0; i<num_layers-1; i++){
        /*
        Flat matrix, row-major: row r = receiving neuron in layer i+1,
        column c = sending neuron in layer i, so element (r, c) lives at
        weights[i][r * sizes[i] + c]. Same shape as np.random.randn(y, x).
        */
        net->weights[i] = malloc(sizes[i] * sizes[i+1] * sizeof(double));
        net->biases[i] = malloc(sizes[i+1] * sizeof(double));
        if (net->weights[i] == NULL || net->biases[i] == NULL) {
            network_destroy(net);
            return NULL;
        }

        for (int j=0; j<sizes[i]*sizes[i+1]; j++){
            net->weights[i][j] = randn();
        }
        for (int j=0; j<sizes[i+1]; j++){
            net->biases[i][j] = randn();
        }
    }

    return net;
}


void network_destroy(Network *net){
    if (net == NULL) {
        return;
    }

    // read num_layers before net is freed
    if (net->weights != NULL) {
        for (int i=0; i<net->num_layers-1; i++){
            free(net->weights[i]);
        }
    }
    free(net->weights);

    if (net->biases != NULL) {
        for (int i=0; i<net->num_layers-1; i++){
            free(net->biases[i]);
        }
    }
    free(net->biases);

    free(net->sizes);
    free(net);
}

/*
Scratch space for backprop
*/
Workspace *workspace_init(const Network *net){
    Workspace *ws = malloc(sizeof(Workspace));
    if(ws == NULL) {
        return NULL;
    }

    memset(ws, 0, sizeof(Workspace)); // so workspace_destroy can destroy a partially build workspace

    ws->num_layers = net->num_layers;

    ws->activations = calloc(net->num_layers, sizeof(double*));
    ws->zs = calloc(net->num_layers - 1, sizeof(double*));
    ws->delta = malloc((size_t)net->max_size * sizeof(double));
    ws->delta_prev = malloc((size_t)net->max_size * sizeof(double));
    ws->y = malloc((size_t)net->sizes[net->num_layers - 1] * sizeof(double));
    if (ws->activations == NULL || ws->zs == NULL || ws->delta == NULL
            || ws->delta_prev == NULL || ws->y == NULL){
        workspace_destroy(ws);
        return NULL;
    }

    for (int l = 0; l < net->num_layers; l++){
        ws->activations[l] = malloc((size_t)net->sizes[l] * sizeof(double));
        if (ws->activations[l] == NULL) {
            workspace_destroy(ws);
            return NULL;
        }
    }
    
    for (int l = 0; l < net->num_layers - 1; l++){
        ws->zs[l] = malloc((size_t)net->sizes[l+1] * sizeof(double));
        if (ws->zs[l] == NULL) {
            workspace_destroy(ws);
            return NULL;
        }
    }

    return ws;
}

void workspace_destroy(Workspace *ws){
    if (ws == NULL){
        return;
    }

    if (ws->activations != NULL){
        for (int l = 0; l < ws->num_layers; l++){
            free(ws->activations[l]);
        }
    }

    free(ws->activations);

    if(ws->zs != NULL){
        for (int l = 0; l < ws->num_layers - 1; l++){
            free(ws->zs[l]);
        }
    }
    free(ws->zs);

    free(ws->delta);
    free(ws->delta_prev);
    free(ws->y);
    free(ws);
}


/*
Dot product replacement function
w = weight matrix
r = row idx
c = column idx
a = incoming activations

https://www.ce.jhu.edu/dalrymple/classes/602/Class12.pdf - page 9, but vector
*/
void matmult(const double *w, const double *a, double *result, int rows, int cols){
    for (int r = 0; r < rows; r++){
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            sum += w[r * cols + c] * a[c];
        }
        result[r] = sum;
    }
}


/*
result = w^T * a, with w stored rows x cols row-major.
a has length rows, result has length cols.
*/
void matmult_T(const double *w, const double *a, double *result, int rows, int cols){
    for (int c = 0; c < cols; c++) {
        result[c] = 0.0;
    }
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            result[c] += w[r * cols + c] * a[r];
        }
    }
}

/*
Forward pass
net = sizes, weights, and biases (read only)
input = sizes[0]
output = sizes[num_layers - 1]
scratch = memory in use, must be 2*max(sizes) (will be imported, dont want to call malloc millions of times)
*/
void feedforward(const Network *net, const double *input, double *output, double *scratch){

    // cur = activiations going into the current layer
    double *cur = scratch;
    // next = activations coming out of current layer
    double *next = scratch + net->max_size;

    for (int i = 0; i < net->sizes[0]; i++){
        cur[i] = input[i];
    }


    for  (int l = 0; l < net->num_layers-1; l++){
        int cols = net->sizes[l];
        int rows = net->sizes[l + 1];

        matmult(net->weights[l], cur, next, rows, cols);


        // add bias
        for (int r = 0; r<rows; r++){
            next[r] = sigmoid(next[r] + net->biases[l][r]);
        }

        // swap pointers
        double *tmp = cur; cur = next; next = tmp; 
    }
    for (int i = 0; i < net->sizes[net ->num_layers -1]; i++){
        output[i] = cur[i];
    }
}

void backprop(const Network *net, const float *x, unsigned char label,
        double **nabla_b, double **nabla_w, Workspace *ws){
    int L = net->num_layers - 2; // idx of the last weight layer
    int classes = net->sizes[net->num_layers - 1];

    // activations[0] = x
    for (int i = 0; i < net->sizes[0]; i++){
        ws->activations[0][i] = x[i];
    }
    
    // forward pass, keeping everything - same loop as feedforward except nothing is overwritten
    // zs[l] is the pre-sigmoid value and activations[l+1] is the post sigmoid one
    for (int l = 0; l <= L; l++){
        int cols = net->sizes[l];
        int rows = net->sizes[l+1];

        matmult(net->weights[l], ws->activations[l], ws->zs[l], rows, cols);
        for (int r = 0; r < rows; r++){
            ws->zs[l][r] += net->biases[l][r];
            ws->activations[l+1][r] = sigmoid(ws->zs[l][r]);
        }
    }

    // the output delta (network.py:102-103)
    
    // equivalent of vectorized_result in mnist_loader.py
    for (int k = 0; k < classes; k++){
        ws->y[k] = 0.0;
    }
    ws->y[label] = 1.0;

    //  delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    cost_derivative(ws->activations[net->num_layers - 1], ws->y, ws->delta, classes);
    for (int r = 0; r < classes; r++){
        ws->delta[r] *= sigmoid_prime(ws->zs[L][r]);
    }

    // backwards loop

    for (int l = L; l >= 0; l--){
        int rows = net->sizes[l + 1];
        int cols = net->sizes[l];

        if (l < L) {
            // delta = (weights[l+1]^T dot delta) * sigmoid_prime(zs[l])
            matmult_T(net->weights[l + 1], ws->delta, ws->delta_prev,
                    net->sizes[l + 2], rows);
            for (int r = 0; r < rows; r++){
                ws->delta_prev[r] *= sigmoid_prime(ws->zs[l][r]);
            }
            double *tmp = ws->delta;
            ws->delta = ws->delta_prev;
            ws->delta_prev = tmp;
        }

        // nabla_b[l] += delta; nabla_w[l] += delta dot activations[l]^T
        for (int r = 0; r < rows; r++){
            nabla_b[l][r] += ws->delta[r];
            for (int c = 0; c < cols; c++){
                nabla_w[l][r * cols + c] += ws->delta[r] * ws->activations[l][c];
            }
        }
    }
}


/*
Number of test inputs the network classifies correctly. The prediction is the
index of the highest activation in the final layer.

pixels = n * sizes[0] floats, row-major, one image per row
labels = n bytes, the correct digit for each row
input   = sizes[0] doubles      \
output  = sizes[num_layers-1]   |  caller-owned, so net never mallocs
scratch = 2 * max_size doubles  /
*/

// could be changed to inline function as it's only called by backprop
void cost_derivative(const double *output_activations, const double *y, double *delta, int n){
	/* modifies delta to be a vector of partial deriviatives 
	\partial C_x /\partial a for the output activations */
	for(int i = 0; i < n; i++) {
		delta[i] = output_activations[i] - y[i];
	}
}

/*
Number of inputs in data the network classifies correctly. The prediction is
the index of the highest activation in the final layer, matching np.argmax in
the reference implementation.

Returns -1 if data does not match the network's input layer, or if the working
buffers could not be allocated.
*/
int evaluate(const Network *net, const Dataset *data){
    int d = net->sizes[0];
    int classes = net->sizes[net->num_layers - 1];
    int correct = 0;

    if (data->d != d){
        fprintf(stderr, "evaluate: data has %d pixels per example, network expects %d\n",
                data->d, d);
        return -1;
    }

    double *input = malloc((size_t)d * sizeof(double));
    double *output = malloc((size_t)classes * sizeof(double));
    double *scratch = malloc(2 * (size_t)net->max_size * sizeof(double));
    if (input == NULL || output == NULL || scratch == NULL){
        fprintf(stderr, "evaluate: out of memory\n");
        free(input); free(output); free(scratch);
        return -1;
    }
    //net for loop can be easily parallelized
    for (int i = 0; i < data->n; i++){
        // the loader stores pixels as float, feedforward wants double (fix later)
        for (int j = 0; j < d; j++){
            input[j] = data->pixels[(size_t)i * d + j];
        }

        feedforward(net, input, output, scratch);

        int best = 0;
        for (int k = 1; k < classes; k++){
            if (output[k] > output[best]){
                best = k;
            }
        }

        if (best == data->labels[i]){
            correct++;
        }
    }

    free(input);
    free(output);
    free(scratch);
    return correct;
}

Grad *grad_init(const Network *net){
    Grad *g = malloc(sizeof(Grad));
    if (g == NULL) return NULL;
    memset(g, 0, sizeof(Grad));
 
    g->num_layers = net->num_layers;
    g->nabla_b = calloc(net->num_layers - 1, sizeof(double*));
    g->nabla_w = calloc(net->num_layers - 1, sizeof(double*));
    if (g->nabla_b == NULL || g->nabla_w == NULL){
        grad_destroy(g);
        return NULL;
    }
 
    for (int l = 0; l < net->num_layers - 1; l++){
        g->nabla_b[l] = malloc((size_t)net->sizes[l+1] * sizeof(double));
        g->nabla_w[l] = malloc((size_t)net->sizes[l] * net->sizes[l+1] * sizeof(double));
        if (g->nabla_b[l] == NULL || g->nabla_w[l] == NULL){
            grad_destroy(g);
            return NULL;
        }
    }
    return g;
}
 
void grad_destroy(Grad *g){
    if (g == NULL) return;
    if (g->nabla_b != NULL)
        for (int l = 0; l < g->num_layers - 1; l++) free(g->nabla_b[l]);
    free(g->nabla_b);
    if (g->nabla_w != NULL)
        for (int l = 0; l < g->num_layers - 1; l++) free(g->nabla_w[l]);
    free(g->nabla_w);
    free(g);
}


 /*Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    is the learning rate.*/
void update_mini_batch(Network *net, const Dataset *data, const int *idx, int m, double eta, Grad *g, Workspace *ws){
    // Zero all calls to prevent batch 2 gradient from stacking on batch 1
    for (int l = 0; l < net->num_layers - 1; l++){
        memset(g->nabla_b[l], 0, (size_t)net->sizes[l+1] * sizeof(double));
        memset(g->nabla_w[l], 0,
               (size_t)net->sizes[l] * net->sizes[l+1] * sizeof(double));
    }

    // for x, y in mini_batch
    for (int i = 0; i < m; i++){
        int e = idx[i];
        const float *x = data->pixels + (size_t)e * data->d;
        backprop(net, x, data->labels[e], g->nabla_b, g->nabla_w, ws);
    }

    // w -= (eta/m) * nabla_w
    // b -= (eta/m) * nabla_b 
    double scale = eta / (double)m;
    for (int l = 0; l < net->num_layers - 1; l++){
        int rows = net->sizes[l+1], cols = net->sizes[l];
        for (int j = 0; j < rows * cols; j++)
            net->weights[l][j] -= scale * g->nabla_w[l][j];
        for (int r = 0; r < rows; r++)
            net->biases[l][r] -= scale * g->nabla_b[l][r];
    }
}


// helper func, replaces random.shuffle() in python
static void shuffle(int *idx, int n){
    for (int i = n - 1; i > 0; i--){
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

int SGD(Network *net, const Dataset *train, int epochs, int mbs, double eta,
        const Dataset *test){

    int n = train->n;

    int *idx = malloc((size_t)n * sizeof(int));
    Workspace *ws = workspace_init(net);
    Grad *g = grad_init(net);
    if (idx == NULL || ws == NULL || g == NULL){
        free(idx); workspace_destroy(ws); grad_destroy(g);
        return -1;
    }
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int e = 0; e < epochs; e++){
        shuffle(idx, n);

        for (int k = 0; k < n; k += mbs){
            int m = (n - k < mbs) ? (n - k) : mbs;
            update_mini_batch(net, train, idx + k, m, eta, g, ws);
        }

        if (test != NULL){
            printf("Epoch %d: %d / %d\n", e, evaluate(net, test), test->n);
        } else {
            printf("Epoch %d complete\n", e);
        }
        fflush(stdout);
    }

    free(idx);
    workspace_destroy(ws);
    grad_destroy(g);
    return 0;
}