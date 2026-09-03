#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    /*
    sizes = number of neurons in respective layer of network

    biases and weights are initialized randomly

    first layer assumed to be input layer
    */
    int num_layers;
    int *sizes;
    int max_size;
    double **biases;
    double **weights;
} Network;

/*
Build a network with num_layers layers, layer i holding sizes[i] neurons.
Weights and biases are drawn from a Gaussian with mean 0 and variance 1.
sizes is copied, so the caller keeps ownership of its array.
Returns NULL if any allocation fails.
*/
Network *network_init(const int *sizes, int num_layers);
void matmult(const double *w, const double *a, double *result, int rows, int cols);

void network_destroy(Network *this);
void network_feedforward(const Network *net, const double *input,
                         double *output, double *scratch);

void cost_derivative(const double *output_activations, const double *y
					 double *delta, int n);
#endif
