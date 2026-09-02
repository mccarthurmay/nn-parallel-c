#include <math.h>

#include "board.h"

// misc functions
float sigmoid(float z){
	// The sigmoid function
	return 1.0f / (1.0f + expf(-x));
}

float sigmoid_prime(float z){
	// Dervative of the sigmoid function
	return sigmoid(z) * (1-sigmoid(z));
}
