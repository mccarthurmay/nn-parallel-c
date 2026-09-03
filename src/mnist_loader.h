#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

extern int n, d;
extern float *pixels;
extern unsigned char *labels;

void load(const char *path);

#endif