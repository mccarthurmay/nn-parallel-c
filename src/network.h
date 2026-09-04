#ifndef NETWORK_H
#define NETWORK_H


#include "mnist_loader.h"
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
void network_destroy(Network *net);
void feedforward(const Network *net, const double *input,
                         double *output, double *scratch);

void cost_derivative(const double *output_activations, const double *y,
					 double *delta, int n);

typedef struct {
    int num_layers;
    double **activations; // num_layers arrays, activations[l] holds sizes[l]
    double **zs;          // num_layers-1 arrays, zs[l] holds sizes[l+1]
    double *delta;        // max_size
    double *delta_prev;   // max_size
    double *y;            // sizes[num_layers-1], the one-hot target  
} Workspace;

Workspace *workspace_init(const Network *net);
void workspace_destroy(Workspace *ws);


// Gradient
typedef struct {
    int num_layers;
    double **nabla_b;   //num_layers-1 arrays, nabla_b[l] holds sizes[l+1] 
    double **nabla_w;   // num_layers-1 arrays, nabla_w[l] holds sizes[l]*sizes[l+1] 
} Grad;


Grad *grad_init(const Network *net);
void grad_destroy(Grad *g);

void update_mini_batch(Network *net, const Dataset *data, const int *idx,
                       int m, double eta, Grad *g, Workspace *ws);


int SGD(Network *net, const Dataset *train, int epochs, int mbs, double eta,
        const Dataset *test);

int evaluate(const Network *net, const Dataset *data);

void backprop(const Network *net, const float *x, unsigned char label,
        double **nabla_b, double **nabla_w, Workspace *ws);

void matmult(const double *w, const double *a, double *result, int rows, int cols);
void matmult_T(const double *w, const double *a, double *result, int rows, int cols);

#endif
