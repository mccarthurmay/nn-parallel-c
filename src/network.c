/*
A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation. 
*/

// Libraries
#include <stdlib.h>

// __init__ clone
typedef struct {
    /*
    sizes = number of neurons in respective layer of network

    biases and weights are initialized randomly

    first layer assumed to be input layer
    */
    int num_layers; 
    int *sizes;
    double **biases;
    double **weights;
} Network;


// Network constructor
Network* network_create(int *sizes, int num_layers){
    Network *net = malloc(sizeof(Network));
    if (net == NULL) {
        printf("network constructor failed\n");
        return NULL;
    }

    net->num_layers = num_layers;
    net->sizes = malloc(num_layers * sizeof(int));
    for (int i=0; i<num_layers; i++){
        net->sizes = sizes[i];
    }

    net->weights = malloc((num_layers -1) * sizeof(double*));
    for (int i=0; i<num_layers-1; i++){
        // input layer doesnt have weights, only output (receiving) layers
        net->weights[i] = malloc(sizes[i] * sizes[i+1] * sizeof(double)); //flat matrix
    }

    net->biases = malloc((num_layers-1) * sizeof(double*));
    for (int i=0; i<num_layers-1; i++){
        net->biases[i] = malloc(sizes[i+1] * sizeof(double)); // input layer doesnt have biases, only output (receiving) layers
    }

    return net;
}

// Gotta fill with random numbers 
/*  self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
*/
