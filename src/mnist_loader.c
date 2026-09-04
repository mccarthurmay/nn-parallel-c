#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "mnist_loader.h"

// "MNST" little-endian, as written by mnist_loader.py
#define MNIST_MAGIC 0x54534E4Du


/*
Load one split written by mnist_loader.py.

File layout: magic, n, d as little-endian uint32, then n*d float32 pixels
(row-major, one image per row), then n uint8 labels.

Returns 0 on success. On failure prints the reason to stderr, returns -1, and
leaves *data zeroed, so a failed load is still safe to hand to dataset_destroy.
*/
int dataset_load(const char *path, Dataset *data){
    uint32_t header[3];
    float *pixels = NULL;
    unsigned char *labels = NULL;
    int n, d;

    // reset values
    data->n = 0;
    data->d = 0;
    data->pixels = NULL;
    data->labels = NULL;

    FILE *f = fopen(path, "rb");
    if (f == NULL){
        fprintf(stderr, "dataset_load: cannot open %s\n", path);
        return -1;
    }

    if (fread(header, sizeof(uint32_t), 3, f) != 3){
        fprintf(stderr, "dataset_load: %s is too short to hold a header\n", path);
        goto fail;
    }
    if (header[0] != MNIST_MAGIC){
        fprintf(stderr, "dataset_load: %s is not an mnist bin (magic %08x)\n",
                path, (unsigned)header[0]);
        goto fail;
    }

    n = (int)header[1];
    d = (int)header[2];
    if (n <= 0 || d <= 0){
        fprintf(stderr, "dataset_load: %s has a bad shape (%d x %d)\n", path, n, d);
        goto fail;
    }

    pixels = malloc((size_t)n * d * sizeof(float));
    labels = malloc((size_t)n);
    if (pixels == NULL || labels == NULL){
        fprintf(stderr, "dataset_load: out of memory for %s (%d x %d)\n", path, n, d);
        goto fail;
    }


    if (fread(pixels, sizeof(float), (size_t)n * d, f) != (size_t)n * d){
        fprintf(stderr, "dataset_load: %s: truncated pixel block\n", path);
        goto fail;
    }
    if (fread(labels, 1, (size_t)n, f) != (size_t)n){
        fprintf(stderr, "dataset_load: %s: truncated label block\n", path);
        goto fail;
    }


    fclose(f);

    data->n = n;
    data->d = d;
    data->pixels = pixels;
    data->labels = labels;
    return 0;
fail:
    free(pixels);
    free(labels);
    fclose(f);
    return -1;
}

void dataset_destroy(Dataset *data){
    if (data == NULL){
        return;
    }
    free(data->pixels);
    free(data->labels);
    data->pixels = NULL;
    data->labels = NULL;
    data->n = 0;
    data->d = 0;
}

