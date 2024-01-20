#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
__global__ void test_build_a(const int N, double* A) {
    A[0] = N;
}

extern "C" {
#include "mc.h"

double* calc_a_matrix(system_t* system) {
    printf("allocating\n");
    double* device_A;
    cudaMalloc(&device_A, 3 * sizeof(double));
    printf("N: %d\n", system->natoms);
    test_build_a<<<1, 1>>>(system->natoms, device_A);
    cudaDeviceSynchronize();

    printf("allocated\n");
    return device_A;
}

void free_a_matrix(double* device_A) {
    printf("Matrix freed\n");
    cudaFree(device_A);
}

}
