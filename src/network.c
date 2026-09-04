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
#include "sigFuncs.h"
#define PI 3.14159265358979323846

/*
Standard normal sample (mean 0, variance 1) via Box-Muller, matching
np.random.randn in the reference implementation. Callers are responsible
for seeding with srand() once at program start.
*/
static double randn(void){
    // shift off the endpoints so u1 is in (0,1) and logf never sees 0
    float u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    float u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
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
    

    this->max_size = 0;
    for (int i = 0; i < num_layers; i++){
        if (sizes[i] > this->max_size){
            this->max_size = sizes[i];
        }
    }
    // input layer doesnt have weights or biases, only output (receiving) layers
    this->weights = calloc(num_layers - 1, sizeof(double*));
    this->biases = calloc(num_layers - 1, sizeof(double*));
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


/*
Number of test inputs the network classifies correctly. The prediction is the
index of the highest activation in the final layer.

pixels = n * sizes[0] floats, row-major, one image per row
labels = n bytes, the correct digit for each row
input   = sizes[0] doubles      \
output  = sizes[num_layers-1]   |  caller-owned, so this never mallocs
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
    int d = this->sizes[0];
    int classes = this->sizes[this->num_layers - 1];
    int correct = 0;

    if (data->d != d){
        fprintf(stderr, "evaluate: data has %d pixels per example, network expects %d\n",
                data->d, d);
        return -1;
    }

    double *input = malloc((size_t)d * sizeof(double));
    double *output = malloc((size_t)classes * sizeof(double));
    double *scratch = malloc(2 * (size_t)net->max_size * sizeof(double));
    
    //this for loop can be easily parallelized
    for (int i = 0; i < data->n; i++){
        // the loader stores pixels as float, feedforward wants double (fix later)
        for (int j = 0; j < d; j++){
            input[j] = data->pixels[(size_t)i * d + j];
        }

        network_feedforward(net, input, output, scratch);

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
