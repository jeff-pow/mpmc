#include <cuda_runtime.h>


// This resulted in the same times across 64, 128, 256, and 512. I just went with a middle ground...
#define THREADS 128
#define MAXFVALUE 1.0e13f
#define halfHBAR 3.81911146e-12     //Ks
#define cHBAR 7.63822291e-12        //Ks //HBAR is already taken to be in Js
#define FINITE_DIFF 0.01            //too small -> vdw calc noises becomes a problem
#define TWOoverHBAR 2.6184101e11    //K^-1 s^-1

__global__ void build_a_matrix(int, double *, const double, double3 *, double *, int);
