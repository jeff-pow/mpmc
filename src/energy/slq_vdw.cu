/**
 * @author Jeff Powell (2023)
 */

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <cuda.h>
#include "defines.h"

// This resulted in the same times across 64, 128, 256, and 512. I just went with a middle ground...
#define THREADS 128
#define MAXFVALUE 1.0e13f
#define halfHBAR 3.81911146e-12     //Ks
#define cHBAR 7.63822291e-12        //Ks //HBAR is already taken to be in Js
#define FINITE_DIFF 0.01            //too small -> vdw calc noises becomes a problem
#define TWOoverHBAR 2.6184101e11    //K^-1 s^-1

__constant__ double basis[9];
__constant__ double recip_basis[9];


/**
 * Method uses exponential polarization regardless of method requested in input
 * script
 */
__global__ static void build_a(int N, double *A, const double damp, double3 *pos, double *pols, const int damp_type) {
    int i = blockIdx.x, j;

    if (i >= N)
        return;

    double r, r2, r3, r5;
    double expr, damping_term1, damping_term2;
    double3 dr, dri, img;

    const double one_over_pol_i = 1.0 / pols[i];
    const double3 pos_i = pos[i];
    const double3 recip_basis_0 =
        make_double3(recip_basis[0], recip_basis[1], recip_basis[2]);
    const double3 recip_basis_1 =
        make_double3(recip_basis[3], recip_basis[4], recip_basis[5]);
    const double3 recip_basis_2 =
        make_double3(recip_basis[6], recip_basis[7], recip_basis[8]);
    const double3 basis_0 = make_double3(basis[0], basis[1], basis[2]);
    const double3 basis_1 = make_double3(basis[3], basis[4], basis[5]);
    const double3 basis_2 = make_double3(basis[6], basis[7], basis[8]);

    const double damp2 = damp * damp;
    const double damp3 = damp2 * damp;

    const int N_per_thread = int(N - 0.5) / THREADS + 1;
    const int threadid = threadIdx.x;
    const int threadid_plus_one = threadIdx.x + 1;
    for (j = threadid * N_per_thread; j < threadid_plus_one * N_per_thread && j < N; j++) {
        if (i == j) {
            A[9 * N * j + 3 * i] = one_over_pol_i;
            A[9 * N * j + 3 * i + 3 * N + 1] = one_over_pol_i;
            A[9 * N * j + 3 * i + 6 * N + 2] = one_over_pol_i;
            A[9 * N * j + 3 * i + 1] = 0.0;
            A[9 * N * j + 3 * i + 2] = 0.0;
            A[9 * N * j + 3 * i + 3 * N] = 0.0;
            A[9 * N * j + 3 * i + 3 * N + 2] = 0.0;
            A[9 * N * j + 3 * i + 6 * N] = 0.0;
            A[9 * N * j + 3 * i + 6 * N + 1] = 0.0;
        } else {
            // START MINIMUM IMAGE
            // get the particle displacement
            dr.x = pos_i.x - pos[j].x;
            dr.y = pos_i.y - pos[j].y;
            dr.z = pos_i.z - pos[j].z;

            // matrix multiply with the inverse basis and round
            img.x = recip_basis_0.x * dr.x + recip_basis_0.y * dr.y +
                recip_basis_0.z * dr.z;
            img.y = recip_basis_1.x * dr.x + recip_basis_1.y * dr.y +
                recip_basis_1.z * dr.z;
            img.z = recip_basis_2.x * dr.x + recip_basis_2.y * dr.y +
                recip_basis_2.z * dr.z;
            img.x = rint(img.x);
            img.y = rint(img.y);
            img.z = rint(img.z);

            // matrix multiply to project back into our basis
            dri.x = basis_0.x * img.x + basis_0.y * img.y + basis_0.z * img.z;
            dri.y = basis_1.x * img.x + basis_1.y * img.y + basis_1.z * img.z;
            dri.z = basis_2.x * img.x + basis_2.y * img.y + basis_2.z * img.z;

            // now correct the displacement
            dri.x = dr.x - dri.x;
            dri.y = dr.y - dri.y;
            dri.z = dr.z - dri.z;
            r2 = dri.x * dri.x + dri.y * dri.y + dri.z * dri.z;

            // various powers of r that we need
            r = sqrt(r2);
            r3 = r2 * r;
            r5 = r3 * r2;
            r3 = 1.0f / r3;
            r5 = 1.0f / r5;
            // END MINIMUM IMAGE

            switch (damp_type) {
                case DAMPING_EXPONENTIAL_UNSCALED: {
                    // damping terms
                    expr = exp(-damp * r);
                    damping_term1 = 1.0f - expr * (0.5f * damp2 * r2 + damp * r + 1.0f);
                    damping_term2 = 1.0f - expr * (damp3 * r * r2 / 6.0f + 0.5f * damp2 * r2 +
                        damp * r + 1.0f);

                    break;
                }
                case DAMPING_AMOEBA: {
                    double l = damp;
                    double u;
                    if (pols[i] * pols[j] == 0) {
                        u = r;
                    } else {
                        u = r / pow(pols[i] * pols[j], 1 / 6.0);
                    }
                    double u3 = u * u * u;
                    double explr = exp(-l * u3);
                    damping_term1 = 1 - explr;
                    damping_term2 = 1 - (1 + l * u3) * explr;

                    break;
                }
                default: { // Damping exponential with corrections
                    double l = damp;
                    double l2 = l * l;
                    double l3 = l * l * l;
                    double u;
                    if (pols[i] * pols[j] == 0) {
                        u = r;
                    } else {
                        u = r / pow(pols[i] * pols[j], 1 / 6.0);
                    }
                    double explr = exp(-l * u);
                    damping_term1 = 1.0 - explr * (.5 * l2 * u * u + l * u + 1.0);
                    damping_term2 = damping_term1 - explr * (l3 * u * u * u / 6.0);

                    break;
                }
            }

            damping_term1 *= r3;
            damping_term2 *= -3.0f * r5;

            // exploit symmetry
            A[9 * N * j + 3 * i] = dri.x * dri.x * damping_term2 + damping_term1;
            const double tmp1 = dri.x * dri.y * damping_term2;
            A[9 * N * j + 3 * i + 1] = tmp1;
            const double tmp2 = dri.x * dri.z * damping_term2;
            A[9 * N * j + 3 * i + 2] = tmp2;
            A[9 * N * j + 3 * i + 3 * N] = tmp1;
            A[9 * N * j + 3 * i + 3 * N + 1] =
                dri.y * dri.y * damping_term2 + damping_term1;
            const double tmp3 = dri.y * dri.z * damping_term2;
            A[9 * N * j + 3 * i + 3 * N + 2] = tmp3;
            A[9 * N * j + 3 * i + 6 * N] = tmp2;
            A[9 * N * j + 3 * i + 6 * N + 1] = tmp3;
            A[9 * N * j + 3 * i + 6 * N + 2] =
                dri.z * dri.z * damping_term2 + damping_term1;
        }
    }
}
extern "C" {
#include <stdlib.h>
#include <time.h>
#include "defines.h"
#include "structs.h"
#include "mc.h"

//only run dsyev from LAPACK if compiled with -llapack, otherwise report an error
#ifdef VDW
//prototype for dsyev (LAPACK)
extern void dsyev_(char *, char *, int *, double *, int *, double *, double *, int *, int *);
#else
void dsyev_(char *a, char *b, int *c, double *d, int *e, double *f, double *g, int *h, int *i) {
    error(
        "ERROR: Not compiled with Linear Algebra VDW.\n");
    die(-1);
}
#endif

static void cudaErrorHandler(cudaError_t error, int line) {
    if (error != cudaSuccess) {
        printf("POLAR_CUDA: GPU is reporting an error: %s %s:%d\n",
               cudaGetErrorString(error), __FILE__, line);
    }
}

void cuda_fast_vdw(void *systemptr) {
    system_t *system = (system_t *)systemptr;
    int N = system->natoms;
    int dim = 3 * N;
    int matrix_size = dim * dim;

    double *host_pols, *host_basis, *host_recip_basis, *host_omegas;
    double3 *host_pos;
    host_pols = (double *)calloc(N, sizeof(double));
    host_pos = (double3 *)calloc(N, sizeof(double3));
    host_basis = (double *)calloc(9, sizeof(double));
    host_recip_basis = (double *)calloc(9, sizeof(double));
    host_omegas = (double *)calloc(N, sizeof(double));

    double *device_pols, *device_A_matrix, *device_omegas, *device_C_matrix;
    double3 *device_pos;
    cudaErrorHandler(cudaMalloc((void **)&device_pols, N * sizeof(double)), __LINE__);
    cudaErrorHandler(cudaMalloc((void **)&device_pos, N * sizeof(double3)), __LINE__);
    cudaErrorHandler(cudaMalloc((void **)&device_A_matrix, matrix_size * sizeof(double)), __LINE__);
    cudaErrorHandler(cudaMalloc((void **)&device_omegas, N * sizeof(double)), __LINE__);
    cudaErrorHandler(cudaMalloc(&device_C_matrix, matrix_size * sizeof(double)), __LINE__);
    cudaDeviceSynchronize();

    // copy over the basis matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            host_basis[i * 3 + j] = (double)system->pbc->basis[j][i];
            host_recip_basis[i * 3 + j] = (double)system->pbc->reciprocal_basis[j][i];
        }
    }
    cudaErrorHandler(cudaMemcpyToSymbol(basis, host_basis, 9 * sizeof(double), 0, cudaMemcpyHostToDevice), __LINE__);
    cudaErrorHandler(cudaMemcpyToSymbol(recip_basis, host_recip_basis, 9 * sizeof(double), 0, cudaMemcpyHostToDevice), __LINE__);
    cudaDeviceSynchronize();

    int i;
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
    for (molecule_ptr = system->molecules, i = 0; molecule_ptr;
    molecule_ptr = molecule_ptr->next) {
        for (atom_ptr = molecule_ptr->atoms; atom_ptr;
        atom_ptr = atom_ptr->next, i++) {
            host_pos[i].x = (double)atom_ptr->pos[0];
            host_pos[i].y = (double)atom_ptr->pos[1];
            host_pos[i].z = (double)atom_ptr->pos[2];
            host_pols[i] = (atom_ptr->polarizability == 0.0)
                ? 1.0f / MAXFVALUE
                : (double)atom_ptr->polarizability;
        }
    }
    cudaErrorHandler(cudaMemcpy(device_pos, host_pos, N * sizeof(double3), cudaMemcpyHostToDevice), __LINE__);
    cudaErrorHandler(cudaMemcpy(device_pols, host_pols, N * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
    cudaDeviceSynchronize();

    build_a<<<N, THREADS>>>(N, device_A_matrix, system->polar_damp, device_pos, device_pols, system->damp_type);
    cudaDeviceSynchronize();
    cudaErrorHandler(cudaGetLastError(), __LINE__ - 1);


    free(host_omegas);
    free(host_basis);
    free(host_recip_basis);
    free(host_pos);
    free(host_pols);
    cudaErrorHandler(cudaFree(device_C_matrix), __LINE__);
    cudaErrorHandler(cudaFree(device_A_matrix), __LINE__);
    cudaErrorHandler(cudaFree(device_omegas), __LINE__);
    cudaErrorHandler(cudaFree(device_pols), __LINE__);
    cudaErrorHandler(cudaFree(device_pos), __LINE__);
    cudaDeviceSynchronize();
}

}
