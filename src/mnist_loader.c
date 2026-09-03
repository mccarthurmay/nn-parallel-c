#include <stdio.h>
#include <stdlib.h>

int n, d;
float *pixels;
unsigned char *labels;

// Load mnist bin info from mnist.pkl.gz (after mnist_loader.py downloads it)
void load(const char *path) {
    unsigned int magic;
    FILE *f = fopen(path, "rb");
    fread(&magic, 4, 1, f);
    fread(&n, 4, 1, f);
    fread(&d, 4, 1, f);
    pixels = malloc((size_t)n * d * sizeof(float));
    labels = malloc(n);
    fread(pixels, sizeof(float), (size_t)n * d, f);
    fread(labels, 1, n, f);
    fclose(f);
}