#include <stdio.h>
#include <stdlib.h>

#include "sigFuncs.h"

int main(int argc, char** argv){

	if (argc > 2){
		fprintf(stderr, "Error: not enough args");
		return 1;
	}

	float val = atof(argv[1]);
	printf("Original value: %f\n", val);
	
	float sigVal = sigmoid(val);
	printf("Sigmoid value: %f\n", sigVal);

	float sigPrime = sigmoid_prime(val);
	printf("Derivative of the sigmoid: %f\n", sigPrime);

	return 0;

}
