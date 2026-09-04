/*
Train a feedforward network on MNIST.

    ./network <epochs> <mini_batch_size> <eta> <hidden>

Run mnist_loader.py once first to produce the .bin files in ../data
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "mnist_loader.h"

#define TRAIN_PATH "../data/train.bin"
#define TEST_PATH  "../data/test.bin"

#define USAGE "usage: %s <epochs> <mini_batch_size> <eta> <hidden>\n" \
              "   eg: %s 30 10 3.0 30\n" \

              // epochs, number of times go through images
              // mini batch size, how many images to look at before updating weights
              // eta, learning rate (3.0)
              // hidden, numbner of neurons in middle layer

int main(int argc, char **argv){

    if (argc != 5){
        fprintf(stderr, USAGE, argv[0], argv[0]);
        return 1;
    }

    int    epochs = atoi(argv[1]);
    int    mbs    = atoi(argv[2]);
    double eta    = atof(argv[3]);
    int    hidden = atoi(argv[4]);

    if (epochs < 1 || mbs < 1 || eta <= 0.0 || hidden < 1){
        fprintf(stderr, "all four must be positive\n");
        fprintf(stderr, USAGE, argv[0], argv[0]);
        return 1;
    }

    // Seed 
    srand(42);

    // dataset_load fills a caller-owned struct 
    Dataset train, test;
    if (dataset_load(TRAIN_PATH, &train) != 0 ||
        dataset_load(TEST_PATH,  &test)  != 0){
        fprintf(stderr, "could not load data -- run mnist_loader.py first\n");
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }

    if (mbs > train.n){
        fprintf(stderr, "mini_batch_size %d exceeds training set (%d)\n", mbs, train.n);
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }

    int sizes[] = { train.d, hidden, 10 };
    Network *net = network_init(sizes, 3);
    if (net == NULL){
        fprintf(stderr, "network_init failed\n");
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }

    printf("net %d-%d-%d   epochs=%d  batch=%d  eta=%.2f\n",
           sizes[0], sizes[1], sizes[2], epochs, mbs, eta);
    printf("train=%d  test=%d\n\n", train.n, test.n);

    clock_t t0 = clock();
    if (SGD(net, &train, epochs, mbs, eta, &test) != 0){
        fprintf(stderr, "SGD failed to allocate\n");
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\ntotal %.2f s   (%.2f s/epoch)\n", secs, secs / epochs);

    network_destroy(net);
    dataset_destroy(&train);
    dataset_destroy(&test);
    return 0;
}