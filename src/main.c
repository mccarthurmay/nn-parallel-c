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
#include "profiling.h"

#define TRAIN_PATH "../data/train.bin"
#define TEST_PATH  "../data/test.bin"
#define USAGE "usage: %s <epochs> <mini_batch_size> <eta> <hidden1> [hidden2 ...]\n" \
              "   eg: %s 30 10 3.0 30        (one hidden layer of 30)\n" \
              "       %s 30 10 3.0 100 30    (two hidden layers)\n"

              // epochs, number of times go through images
              // mini batch size, how many images to look at before updating weights
              // eta, learning rate (3.0)
              // hidden, numbner of neurons in middle layer

int main(int argc, char **argv){
    PROF_INIT(argc, argv);
    PROF_PHASE(p_load, "Phase_Load");
    PROF_PHASE(p_init, "Phase_Init");
    PROF_PHASE(p_train, "Phase_Train");


    if (argc < 5){
        fprintf(stderr, USAGE, argv[0], argv[0]);
        return 1;
    }

    int    epochs = atoi(argv[1]);
    int    mbs    = atoi(argv[2]);
    double eta    = atof(argv[3]);
    int num_hidden=argc - 4;

    if (epochs < 1 || mbs < 1 || eta <= 0.0 || hidden < 1){
        fprintf(stderr, "all four must be positive\n");
        fprintf(stderr, USAGE, argv[0], argv[0]);
        return 1;
    }
    for (int i = 0; i < num_hidden; i++){
        if (atoi(argv[4 + i])< 1){
            fprintf(stderr, "hidden layer %d must be positive\n", i + 1);
            return 1;
        }

    // Seed 
    srand(42);

    // dataset_load fills a caller-owned struct 
    PROF_PHASE_START(p_load);
    Dataset train, test;
    if (dataset_load(TRAIN_PATH, &train) != 0 ||
        dataset_load(TEST_PATH,  &test)  != 0){
        fprintf(stderr, "could not load data -- run mnist_loader.py first\n");
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }
    PROF_PHASE_STOP(p_load);

    if (mbs > train.n){
        fprintf(stderr, "mini_batch_size %d exceeds training set (%d)\n", mbs, train.n);
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }
    
    PROF_PHASE_START(p_init);
    int num_layers = num_hidden + 2;          // input + hidden layers + output
    int *sizes = malloc((size_t)num_layers * sizeof(int));
    if (sizes == NULL){
        fprintf(stderr, "out of memory\n");
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }
    sizes[0] = train.d;                        // 784 pixels
    for (int i = 0; i < num_hidden; i++){
        sizes[i + 1] = atoi(argv[4 + i]);
    }
    sizes[num_layers - 1] = 10;                // ten digits

    Network *net = network_init(sizes, num_layers);
    PROF_PHASE_STOP(p_init);
    if (net == NULL){
        fprintf(stderr, "network_init failed\n");
        dataset_destroy(&train); dataset_destroy(&test);
        return 1;
    }

    printf("net ");
    for (int l = 0; l < num_layers; l++){
        printf("%d%s", sizes[l], (l == num_layers - 1) ? "" : "-");
    }
    printf("   epochs=%d  batch=%d  eta=%.2f\n", epochs, mbs, eta);
    printf("train=%d  test=%d\n\n", train.n, test.n);
    clock_t t0 = clock();

    PROF_PHASE_START(p_train);
    if (SGD(net, &train, epochs, mbs, eta, &test) != 0){
        fprintf(stderr, "SGD failed to allocate\n");
    }
    PROF_PHASE_STOP(p_train);

    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\ntotal %.2f s   (%.2f s/epoch)\n", secs, secs / epochs);

    network_destroy(net);
    free(sizes);
    dataset_destroy(&train);
    dataset_destroy(&test);
    return 0;
}
