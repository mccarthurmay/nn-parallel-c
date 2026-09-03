#include <math.h>

#include "sigFuncs.h"

// misc functions
double sigmoid(double z){
	// The sigmoid function
	return 1.0 / (1.0 + exp(-z));
}

double sigmoid_prime(double z){
	// Dervative of the sigmoid function
	return sigmoid(z) * (1-sigmoid(z));
}
