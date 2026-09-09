/*
A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation.
*/

// Libraries
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "networkV2.h"
#include "mnist_loader.h"
#include "sigFuncs.h"
#include "profiling.h"
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

/*
Build a network with num_layers layers 

Layout:
num_layers  int, number of layers in sizes[] array

sizes       array, a copy of the caller's array, num_layers ints long. It is copied rather than storing the caller's pointer
            to retain validity if freed or reused by caller. Also prevents future collisions when in parallel. 
            holds an array of the layers, such as {784, 30, 10} (784 neurons in inpuht layer, 30 in hidden layer, 10 in output layer)

max_size    largest number in sizes array. Other functions need this number to allocate 'scratch' buffer memory to prevent memory overflow

weights     array, of num_layers-1 pointers. weights[i] points at one block of doubles holding every connection from layer i to layer i+1. 
            Each block is sizes[i] * sizes[i+1] doubles long, representing a grid of sizes[i+1] rows by sizes[i] columns where:
                row r = a receiving neuron in layer i +1
                column c = a sending neuron in layer i
            
            The grid is stored flat, in row-major order. 
                weights[i][r * sizes[i] + c]
            
biases      array, of num_layers-1 pointers. biases[i] points at sizes[i+1] doubles. 

Example sizes = {784, 30, 10}:
max_sizes   784
weights[0]  784 * 30 = 23520 doubles    grid: 30 rows x 784 columns
biases[0]   30 doubles
weights[1]  30 * 10 = 300 doubles       grid: 10 rows x 30 columns
biases[1]   10 doubles
*/
Network *network_init(const int *sizes, int num_layers){
    Network *net = malloc(sizeof(Network));

    if (net == NULL) {
        return NULL;
    }

    // zeroed up front so network_destroy can clean up a partly built network
    memset(net, 0, sizeof(Network));

    net->num_layers = num_layers;
    // create own copy of sizes array. makes network independent
    net->sizes = malloc(num_layers * sizeof(int));

    if (net->sizes == NULL) {
        network_destroy(net);
        return NULL;
    }

    // copy bytes from original sizes array to our own sizes array
    memcpy(net->sizes, sizes, num_layers * sizeof(int));
    
    // Get largest layer, used for scratch buffer later 
    net->max_size = 0;
    for (int i = 0; i < num_layers; i++){
        if (sizes[i] > net->max_size){
            net->max_size = sizes[i];
        }
    }

    // input layer doesnt have weights or biases, only output (receiving) layers hence num_layers -1
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

/*
Frees network's location in memory. 
*/
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
Allocates scratch space that backprop needs, returning a pointer to it

Layout:
num_layers      int. copied from network

activations     double**. activations[l] is an array for layer l. activations[l][i] is neuron i's value in that layer, after sigmoid is applied.

zs              double*, zs[l] holds sizes[l+1] doubles, value before sigmoid is applied 

delta           double*, flat array of max_size doubles. 

delta_prev      double*, flat array, same size as delta

y               double*, flat array of sizes[num_layers-1] doubles, always 10 for mnist (10 digits). All zeros except for 1.0 at the correct digit.


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

    // fill in activations list
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

/*
Frees workspace memory
*/
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
Scratch for batch_backprop, sized for up to max_batch examples at once.
A[l] is max_batch x sizes[l]; Z[l] and D[l] are max_batch x sizes[l+1].
Returns NULL if any allocation fails.
*/
BatchWorkspace *batch_workspace_init(const Network *net, int max_batch){
    BatchWorkspace *ws = malloc(sizeof(BatchWorkspace));
    if (ws == NULL) return NULL;
    memset(ws, 0, sizeof(BatchWorkspace));

    ws->num_layers = net->num_layers;
    ws->max_batch = max_batch;

    ws->A = calloc(net->num_layers, sizeof(double*));
    ws->Z = calloc(net->num_layers - 1, sizeof(double*));
    ws->D = calloc(net->num_layers - 1, sizeof(double*));
    if (ws->A == NULL || ws->Z == NULL || ws->D == NULL){
        batch_workspace_destroy(ws);
        return NULL;
    }

    for (int l = 0; l < net->num_layers; l++){
        ws->A[l] = malloc((size_t)max_batch * net->sizes[l] * sizeof(double));
        if (ws->A[l] == NULL){ batch_workspace_destroy(ws); return NULL; }
    }
    for (int l = 0; l < net->num_layers - 1; l++){
        ws->Z[l] = malloc((size_t)max_batch * net->sizes[l+1] * sizeof(double));
        ws->D[l] = malloc((size_t)max_batch * net->sizes[l+1] * sizeof(double));
        if (ws->Z[l] == NULL || ws->D[l] == NULL){
            batch_workspace_destroy(ws);
            return NULL;
        }
    }
    return ws;

}

void batch_workspace_destroy(BatchWorkspace *ws){
    if (ws == NULL) return;
    if (ws->A != NULL){
        for (int i = 0; i < ws->num_layers; i++) free(ws->A[i]);
    }
    free(ws->A);
    if(ws->Z != NULL) {
        for (int i = 0; i < ws->num_layers - 1; i++) free(ws->Z[i]);
    }
    free(ws->Z);
    if (ws->D != NULL) {
        for (int i = 0; i < ws->num_layers - 1; i++) free(ws->D[i]);
    }
    free(ws->D);
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
Runs one image through the network and writes out its 10 output values

Computes two operations, reapeated once per layer transition:

z = W dot a + b     every neuron's weighted sum, plus its bias
a = sigmoid(z)      make each total range 0 to 1

output of one layer is the input of the next layer. 

function only reads weights and biases. Used during training

Params:
net         weights, biases,  and layer sizes (read only)

input       double*, sizes[0] doubles. input is the image. read only.

output      double*, sizes[num_layers-1] doubles, always 10 for mnist. 

scratch     2*max_sizes doubles
*/
void feedforward(const Network *net, const double *input, double *output, double *scratch){
    // cur = activiations going into the current layer
    // points at first half of scratch block
    double *cur = scratch;
    // next = activations coming out of current layer
    // points at next half
    double *next = scratch + net->max_size;

    //copy image into cur, first layer
    for (int i = 0; i < net->sizes[0]; i++){
        cur[i] = input[i];
    }

    for  (int l = 0; l < net->num_layers-1; l++){
        int cols = net->sizes[l];
        int rows = net->sizes[l + 1];

        // get weighted sums
        matmult(net->weights[l], cur, next, rows, cols);


        // add bias
        for (int r = 0; r<rows; r++){
            // next gets the finished activations for receiving layer
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
Computes gradient for one trading image, adding it to nabla_b/nabla_w

Training is essentially just nudging every weight and bias a little in a direction that makes the network less wrong. 
In order to do this, each individual parameter needs a partial derivative, or a number that indicates the "cost" for every "nudge" on the parameter. 
A collection of partial derivatives is a gradient, represented by the symbol nabla. 

Delta[r] is the error at neuron r, or the derivative of the cost with respect to that neuron's pre-sigmoid total z. 
        gradient for bias r = delta[r]
        gradient for weight (r, c) = delta [r] * activation of sending neuron c

Params:

net         Network *,weights, biases, sizes. Read only.

x           float *, the image

label       char, the correct digit (0-9)

nabla_b     double **, caller's gradient accumulators

nabla_w     double **, biases and weights

ws          workspace *, scratch space allocated by SGD

*/
void backprop(const Network *net, const float *x, unsigned char label,
        double **nabla_b, double **nabla_w, Workspace *ws){
    int L = net->num_layers - 2; // idx of the last weight layer
    int classes = net->sizes[net->num_layers - 1];

    // activations[0] = x
    // load image into input layer
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
            // adds bias into z
            ws->zs[l][r] += net->biases[l][r];
            // 0-1 value into activations
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
    //
    //  Two factors multiplied together
    //
    //  First factor - how wrong the answer was (cost)
    cost_derivative(ws->activations[net->num_layers - 1], ws->y, ws->delta, classes);
    //  Second factor - how responsive the neuron was.
    //  if confidently wrong (z far from 0), get a small delta and no learning
    for (int r = 0; r < classes; r++){
        ws->delta[r] *= sigmoid_prime(ws->zs[L][r]);
    }

    // backwards loop
    //
    // walks layers from last to first. push delta back one layer then convert current delta into gradients
    for (int l = L; l >= 0; l--){
        int rows = net->sizes[l + 1];
        int cols = net->sizes[l];

        if (l < L) {
            // delta = (weights[l+1]^T dot delta) * sigmoid_prime(zs[l])
            // 
            // sends error back through connections. 
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

Essentially counts how many images in a dataset the network classifies correctly

Params:
net         const Network *
            Pointer to one Network struct. Read only.

data        const Dataset *
            Pointer to one Dataset struct. Read only. 
            data->n         int, number of images
            data->d         int, pixels per image, must match sizes[0]
            data->pixels    float*, ONE flat array of n * d floats, all images end to end, row-major
            data-> labels   unsigned char*,  flat array of n bytes, the correct digit for each image

Returns int, number of correct classificiations
*/
int evaluate(const Network *net, const Dataset *data){
    int d = net->sizes[0];
    int classes = net->sizes[net->num_layers - 1];

    // Tally of correct numbers
    int correct = 0;

    if (data->d != d){
        fprintf(stderr, "evaluate: data has %d pixels per example, network expects %d\n",
                data->d, d);
        return -1;
    }

    //3 Working buffers
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

/*
Allocates the gradient accumulators for one mini-batch. Returns a pointer to allocated space

A grad is shaped like a network's wweights and biases, but it just holds gradients rather than parameters.
For every weight in the network, there is a number saying which direction the weight should move and how strongly
it should move in that direction. 
        net -> weights[l][i]       actual weight stored in network
        g -> nabla_w[l][i]          how much to change that weight

Param:
net         const Network *
            Pointer to one Network struct. Read-only. Uses shape (num_layers and sizes[])


Layout:
    num_layers      int. Copied from net.

    nabla_b         double**. has num_layers-1 pointers that are each given a block of sizes[l+1] doubles, giving one gradient per receiving neuron. 
                    nabla_b[l][r]
    
    nabla_w         double**. num_layers-1 pointers that are each given a block of sizes[l] * sizes[l+1] doubles. Uses same row-major layout as network_init

        nabla_b[0]   30 doubles         matches biases[0]
        nabla_b[1]   10 doubles         matches biases[1]
        nabla_w[0]   23,520 doubles     matches weights[0], 30 x 784
        nabla_w[1]   300 doubles        matches weights[1], 10 x 30

*/
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
 

/*
Clean up grad
*/
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


/*
Runs one mini-batch and applies the result to the network's weights and biases. Where the "learning" happens. 

1. Zero the accumulators
2. Call the backprop once per image in the batch, summing the gradients
3. Subtract teh averaged gradient from every parameter

- Gradient descent

Params:
net         Network *
            Not set as a constant, gets modified

data        const Dataset *
            read-only. Receives whole dataset plus a list of images to use.

idx         const int *
            read-only. A pointer into the middle of SGD's shuffled index array. 

m           int
            number of images in this batch. 

eta         double
            learning rate, step size. If too small, training becomes slow. If too large, inaccurate. 

g           Grad *
            Accumulators, allocated by SGD and reused each batch.

ws          Workspace *
            Backprop's scratch, allocated once by SGD. 
    
*/
void update_mini_batch(Network *net, const Dataset *data, const int *idx, int m, double eta, Grad *g, BatchWorkspace *ws){
    // Zero all calls to prevent batch 2 gradient from stacking on batch 1
    for (int l = 0; l < net->num_layers - 1; l++){
        memset(g->nabla_b[l], 0, (size_t)net->sizes[l+1] * sizeof(double));
        memset(g->nabla_w[l], 0,
               (size_t)net->sizes[l] * net->sizes[l+1] * sizeof(double));
    }


    // one call for the whole batch, replacing the m separate backprop calls
    batch_backprop(net, data, idx, m, g->nabla_b, g->nabla_w, bws);

    // w -= (eta/m) * nabla_w
    // b -= (eta/m) * nabla_b 

    // 3. Subtract teh averaged gradient from every parameter
    //
    // dividing by m    turns the sum from the above loop into an average
    // 
    // multiply by eta  scales the average down
    double scale = eta / (double)m;

    // walk the grid and subtract averaged gradient for weights and biases
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

/*
Trains network by stochastic (steps are based on small, randomly chosen batches) gradient descent, for a given number of epochs

For each epoch, shuffle the training set, break into mini-batches, and run update_mini_batch on the batches. 

Params:
net         Network *
            Gets modified in update_mini_batch

train       const Dataset *
            Read-only.

epochs      int
            How many complete passes over training set.

mbs         int
            Mini-batch size, images per step

eta         double 
            Learning rate, step size

test        const Dataset *
            read-only. REMOVE IF PER_EPOCH ACCURACYT IS NOT IMPORTANT

returns int 0 on success, -1 if scratch allocation fails.
*/
int SGD(Network *net, const Dataset *train, int epochs, int mbs, double eta,
        const Dataset *test){
    PROF_PHASE(p_epoch, "Phase_TrainEpoch");
    PROF_PHASE(p_eval, "Phase_Evaluate");
    

    int n = train->n;

    //idx   n ints, one per training image. holds the order images are visted in epoch. Filled and reshuffled for each epoch
    int *idx = malloc((size_t)n * sizeof(int));
    //ws    backprop's per image scratch
    BatchWorkspace *ws = batch_workspace_init(net, mbs);
    //g     per batch gradient accumulators. 
    Grad *g = grad_init(net);



    if (idx == NULL || ws == NULL || g == NULL){
        free(idx); batch_workspace_destroy(ws); grad_destroy(g);
        return -1;
    }

    for (int i = 0; i < n; i++) idx[i] = i;

    // One iteration per epoch
    for (int e = 0; e < epochs; e++){
        shuffle(idx, n);
        PROF_PHASE_START(p_epoch);
        // Walk through shuffled order in chunks of mbs
        for (int k = 0; k < n; k += mbs){

            // Checks if batch == mbs, last batch is often short
            //
            //
            // COULD BE OPtIMIZED, ONLY RAN ON LAST BATCH
            //
            //
            int m = (n - k < mbs) ? (n - k) : mbs;

            
            update_mini_batch(net, train, idx + k, m, eta, g, ws);
        }
        PROF_PHASE_STOP(p_epoch);

        if (test != NULL){
            PROF_PHASE_START(p_eval);
            int correct = evaluate(net, test);
            PROF_PHASE_STOP(p_eval);
            printf("Epoch %d: %d / %d\n", e, correct, test->n);
        } else {
            printf("Epoch %d complete\n", e);
        }
        fflush(stdout);
    }

    free(idx);
    batch_workspace_destroy(ws);
    grad_destroy(g);
    return 0;
}



/*
Z = A * W^T with bias, then activations. A is m x cols (one example per row),
W is rows x cols row-major, Z and Aout are m x rows.

r is the outer loop so W[r] stays in L1 across all m examples 
*/
static void batch_forward(const double *A, const double *W, const double *b,
        double *Z, double *Aout, int m, int rows, int cols){
    for (int r = 0; r < rows; r++){
        const double *w = W + (size_t)r * cols;
        double br = b[r];
        for (int i = 0; i < m; i++) {
            const double *a = A + (size_t)i * cols;
            double sum = 0.0;
            for (int c = 0; c < cols; c++){
                sum += w[c] * a[c];
            }
            double zv = sum + br;
            Z[(size_t)i * rows + r] = zv;
            Aout[(size_t)i * rows + r] = sigmoid(zv);
        }
    }
}



/*
Dprev = (D * W) elementwise-times sigmoid'(z) for the previous layer.
D is m x rows, W is rows x cols, Dprev and Aprev are m x cols.

sigmoid'(z) is computed as a*(1-a) from the stored activation, which avoids
two exp() calls per neuron -- sigmoid_prime was ~3% of the profiled run.
*/
static void batch_backward(const double *D, const double *W, const double *Aprev,
        double *Dprev, int m, int rows, int cols){

    for (int i = 0; i < m; i++){
        double *dp = Dprev + (size_t)i * cols; 
        const double *d = D + (size_t)i * rows;

        for (int c = 0; c < cols; c++) dp[c] = 0.0;

        for (int r = 0; r < rows; r++){
            double dr = d[r];
            const double *w = W + (size_t)r * cols;
            for (int c = 0; c < cols; c++){
                dp[c] += dr * w[c];
            }
        }

        const double *ap = Aprev + (size_t)i * cols;
        for (int c = 0; c < cols; c++){
            dp[c] *= ap[c] * (1.0 - ap[c]);
        }
    }
}

/*
nabla_w += D^T * A, nabla_b += column sums of D.
D is m x rows, A is m x cols, nabla_w is rows x cols.

One row of nabla_w (cols doubles) stays in L1 while all m examples accumulate
into it, so the full gradient array is walked once per mini-batch instead of
once per example.
*/

static void batch_grad(const double *D, const double *A,
        double *nabla_w, double *nabla_b,
        int m, int rows, int cols){
    for (int r = 0; r < rows; r++){
        double *nw = nabla_w + (size_t)r * cols;
        double nb = 0.0;
        for (int i = 0; i < m; i++) {
            double dr = D[(size_t)i * rows + r];
            nb += dr;
            const double *a = A + (size_t)i * cols;
            for (int c = 0; c < cols; c++){
                nw[c] += dr * a[c];
            }
        }
        nabla_b[r] += nb;
    }
}

/*
batch_backprop replaces the m separate backprop calls in update_mini_batch
*/

void batch_backprop(const Network *net, const Dataset *data, const int *idx, int m,
        double **nabla_b, double **nabla_w, BatchWorkspace *ws){
    int L = net->num_layers - 2;
    int d = net->sizes[0];
    int classes = net->sizes[net->num_layers - 1];

    // gather m examples into A[0], widening float -> double
    for (int i = 0; i < m; i++){
        const float *px = data->pixels + (size_t)idx[i] * d;
        double *a = ws->A[0] + (size_t)i * d;
        for (int j = 0; j < d; j++) a[j] = px[j];
    }

    for (int l = 0; l <= L; l++){
        batch_forward(ws->A[l], net->weights[l], net->biases[l],
                ws->Z[l], ws->A[l + 1],
                m, net->sizes[l + 1], net->sizes[l]);
    }
    // output delta for all m at once: (a - y) * a * (1 - a)
    for (int i = 0; i < m; i ++){
        const double *a = ws->A[net->num_layers - 1] + (size_t)i * classes;
        double *dl = ws->D[L] + (size_t)i * classes;
        unsigned char label = data->labels[idx[i]];
        for (int k = 0; k < classes; k++){
            double y = (k == label) ? 1.0 : 0.0;
            dl[k] = (a[k] - y) * a[k] * (1.0 - a[k]);
        }
    }

    for (int l = L; l >= 0; l--){
        if (l < L){
            batch_backward(ws->D[l + 1], net->weights[l + 1], ws->A[l + 1],
                    ws->D[l], m, net->sizes[l + 2], net->sizes[l + 1]);
        }
        batch_grad(ws->D[l], ws->A[l], nabla_w[l], nabla_b[l],
                m, net->sizes[l + 1], net->sizes[l]);
    }
}


