#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

typedef struct {
    int n; // number of examples
    int d; // pixels per example
    float *pixels; // n * d, row-major
    unsigned char *labels; // n
} Dataset;

int dataset_load(const char *path, Dataset *data);
void dataset_destroy(Dataset *data);

#endif
