/*
A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation.
*/

// Libraries
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"

/*
Standard normal sample (mean 0, variance 1) via Box-Muller, matching
np.random.randn in the reference implementation. Callers are responsible
for seeding with srand() once at program start.
*/
static float randn(void){
    // shift off the endpoints so u1 is in (0,1) and logf never sees 0
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// Network constructor
Network *network_init(const int *sizes, int num_layers){
    Network *this = malloc(sizeof(Network));
    if (this == NULL) {
        return NULL;
    }
    // zeroed up front so network_destroy can clean up a partly built network
    memset(this, 0, sizeof(Network));

    this->num_layers = num_layers;
    this->sizes = malloc(num_layers * sizeof(int));
    if (this->sizes == NULL) {
        network_destroy(this);
        return NULL;
    }
    memcpy(this->sizes, sizes, num_layers * sizeof(int));

    // input layer doesnt have weights or biases, only output (receiving) layers
    this->weights = calloc(num_layers - 1, sizeof(float*));
    this->biases = calloc(num_layers - 1, sizeof(float*));
    if (this->weights == NULL || this->biases == NULL) {
        network_destroy(this);
        return NULL;
    }

    for (int i=0; i<num_layers-1; i++){
        /*
        Flat matrix, row-major: row r = receiving neuron in layer i+1,
        column c = sending neuron in layer i, so element (r, c) lives at
        weights[i][r * sizes[i] + c]. Same shape as np.random.randn(y, x).
        */
        this->weights[i] = malloc(sizes[i] * sizes[i+1] * sizeof(float));
        this->biases[i] = malloc(sizes[i+1] * sizeof(float));
        if (this->weights[i] == NULL || this->biases[i] == NULL) {
            network_destroy(this);
            return NULL;
        }

        for (int j=0; j<sizes[i]*sizes[i+1]; j++){
            this->weights[i][j] = randn();
        }
        for (int j=0; j<sizes[i+1]; j++){
            this->biases[i][j] = randn();
        }
    }

    return this;
}


void network_destroy(Network *this){
    if (this == NULL) {
        return;
    }

    // read num_layers before this is freed
    if (this->weights != NULL) {
        for (int i=0; i<this->num_layers-1; i++){
            free(this->weights[i]);
        }
    }
    free(this->weights);

    if (this->biases != NULL) {
        for (int i=0; i<this->num_layers-1; i++){
            free(this->biases[i]);
        }
    }
    free(this->biases);

    free(this->sizes);
    free(this);
}

/*
Dot product replacement function
w = weight matrix
r = row idx
c = column idx
a = incoming activations

https://www.ce.jhu.edu/dalrymple/classes/602/Class12.pdf - page 9, but vector
*/
void matmult(const double *w, const double *a, double *result, int rows, int cols)?{
    for (int r = 0; r < rows; r++){
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            sum += w[r * cols + c] * a[c];
        }
        result[r] = sum
    }
}


/*
Forward pass
net = sizes, weights, and biases (read only)
input = sizes[0]
output = sizes[num_layers - 1]
scratch = memory in use, must be 2*max(sizes) (will be imported, dont want to call malloc millions of times)
*/
void network_feedforward(Network *net, double *input, double *output, double *scratch){
    // Compute largest layer for scratch
    int maxn = 0;
    for (int i = 0; i < net->num_layers; i++){
        if (net->sizes[i]>maxn){
            maxn = net->sizes[i];
        }
    }

    // cur = activiations going into the current layer
    double *cur = scratch
    // next = activations coming out of current layer
    double *next = scratch + maxn;

    for (int i = 0; i < net->sizes[0]; i++){
        cur[i] = input[i];
    }


    for  (int l = 0; l < net->num_layers-1; l++){
        int cols = net->sizes[l]
        int rows = net->sizes[l + 1];

        matmult(net->weights[l], cur, next, rows, cols);


        // add bias
        for (int r = 0; r<rows; r++){
            next[r] = sigmoid(next[r] + net->biases[l][r]);
        }

        // swap pointers
        double *tmp = cur; cur = next; next = tmp; 
    }
    for (int i = 0; i < net->sizes[net ->num_layers -1]; i++;){
        output[i] = cur[i];
    }
}