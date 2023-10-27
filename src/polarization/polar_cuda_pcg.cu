/*

   @2018, Adam Hogan
   @2010, Jonathan Belof
   University of South Florida

 */

#include "cublas_v2.h"
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <defines.h>
#include "cuda_functions.h"




__global__ void precondition_z(int N, double *A, double *r, double *z) {
    int i = blockIdx.x;
    if (i < N)
        z[i] = 1.0 / A[N * i + i] * r[i];
    return;
}

__global__ void print_b(int N, double *B) {
    for (int i = 0; i < 3; i++) {
        printf("a");
    }
}


__global__ static void print_a(int N, double *A) {
    printf("N: %d\n", N);
    for (int i = 0; i < 3 * 3 * N * N; i++) {
        printf("%8.5f ", A[i]);
        if ((i + 1) % (N * 3) == 0 && i != 0) {
            printf("\n");
        }
    }
    printf("\n");
}


extern "C" {

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <structs.h>
#include <time.h>
#include "defines.h"
#include "structs.h"
#include "mc.h"
#include "cuda_functions.h"

    void thole_field(system_t *);

    int getSPcores(cudaDeviceProp devProp) {
        int cores = 0;
        int mp = devProp.multiProcessorCount;
        switch (devProp.major) {
            case 2: // Fermi
                if (devProp.minor == 1)
                    cores = mp * 48;
                else
                    cores = mp * 32;
                break;
            case 3: // Kepler
                cores = mp * 192;
                break;
            case 5: // Maxwell
                cores = mp * 128;
                break;
            case 6: // Pascal
                if (devProp.minor == 1)
                    cores = mp * 128;
                else if (devProp.minor == 0)
                    cores = mp * 64;
                else
                    printf("Unknown device type\n");
                break;
            case 7: // Volta
                if (devProp.minor == 0)
                    cores = mp * 64;
                else
                    printf("Unknown device type\n");
                break;
            default:
                printf("Unknown device type\n");
                break;
        }
        return cores;
    }

    static const char *cublasGetErrorEnum(cublasStatus_t error) {
        switch (error) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";

            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";

            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";

            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";

            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";

            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";

            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";

            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";

            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";

            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
        }

        return "<unknown>";
    }

    static void cudaErrorHandler(cudaError_t error, int line) {
        if (error != cudaSuccess) {
            printf("POLAR_CUDA: GPU is reporting an error: %s %s:%d\n",
                    cudaGetErrorString(error), __FILE__, line);
        }
    }

    void cublasErrorHandler(cublasStatus_t error, int line) {
        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("POLAR_CUDA: CUBLAS is reporting an error: %s %s:%d\n",
                    cublasGetErrorEnum(error), __FILE__, line);
        }
    }

    void *polar_cuda(void *ptr) {
        system_t *system = (system_t *)ptr;
        molecule_t *molecule_ptr;
        atom_t *atom_ptr;
        int i, j, iterations;
        double potential = 0.0;
        double alpha, beta, result;
        int N = system->natoms;
        double *host_x, *host_b, *host_basis, *host_recip_basis,
              *host_pols; // host vectors
        double3 *host_pos;
        double *A;                                            // GPU matrix
        double *x, *r, *z, *p, *tmp, *r_prev, *z_prev, *pols; // GPU vectors
        double3 *pos;
        const double one = 1.0; // these are for some CUBLAS calls
        const double zero = 0.0;
        const double neg_one = -1.0;

        cudaError_t error;     // GetDevice and cudaMalloc errors
        cudaDeviceProp prop;   // GetDevice properties
        cublasHandle_t handle; // CUBLAS handle
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            cudaErrorHandler(error, __LINE__);
            return NULL;
        } else {
            if (system->step == 0) {
                printf(
                        "POLAR_CUDA: Found %s with pci bus id %d, %d MB and %d cuda cores\n",
                        prop.name, prop.pciBusID, (int)prop.totalGlobalMem / 1000000,
                        getSPcores(prop));
            }
        }

        cublasErrorHandler(cublasCreate(&handle),
                __LINE__); // initialize CUBLAS context

        host_b = (double *)calloc(3 * N, sizeof(double)); // allocate all our arrays
        host_x = (double *)calloc(3 * N, sizeof(double));
        host_basis = (double *)calloc(9, sizeof(double));
        host_recip_basis = (double *)calloc(9, sizeof(double));
        host_pos = (double3 *)calloc(N, sizeof(double3));
        host_pols = (double *)calloc(N, sizeof(double));

        cudaErrorHandler(cudaMalloc((void **)&x, 3 * N * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&A, 3 * N * 3 * N * sizeof(double)),
                __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&r, 3 * N * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&z, 3 * N * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&p, 3 * N * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&tmp, 3 * N * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&r_prev, 3 * N * sizeof(double)),
                __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&z_prev, 3 * N * sizeof(double)),
                __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&pos, N * sizeof(double3)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&pols, N * sizeof(double)), __LINE__);

        // copy over the basis matrix
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                host_basis[i * 3 + j] = (double)system->pbc->basis[j][i];
                host_recip_basis[i * 3 + j] = (double)system->pbc->reciprocal_basis[j][i];
            }
        }


        thole_field(system); // calc static e-field

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
                for (j = 0; j < 3; j++) {
                    host_b[3 * i + j] =
                        (double)(atom_ptr->ef_static[j] + atom_ptr->ef_static_self[j]);
                    host_x[3 * i + j] = (double)system->polar_gamma *
                        atom_ptr->polarizability * host_b[3 * i + j];
                }
            }
        }

        cudaErrorHandler(
                cudaMemcpy(pos, host_pos, N * sizeof(double3), cudaMemcpyHostToDevice),
                __LINE__); // copy over pos (to pos), b (to r), x (to x) and pols (to
                           // pols)
        cudaErrorHandler(
                cudaMemcpy(r, host_b, 3 * N * sizeof(double), cudaMemcpyHostToDevice),
                __LINE__);
        cudaErrorHandler(
                cudaMemcpy(x, host_x, 3 * N * sizeof(double), cudaMemcpyHostToDevice),
                __LINE__);
        cudaErrorHandler(
                cudaMemcpy(pols, host_pols, N * sizeof(double), cudaMemcpyHostToDevice),
                __LINE__);

        // make A matrix on GPU
        // build_a_matrix(system);
        //build_a_matrix<<<N, THREADS>>>(N, A, system->polar_damp, pos, pols, system->damp_type);
        A = init_A_matrix(system);
        cudaErrorHandler(cudaGetLastError(), __LINE__ - 1);

        // R = B - A*X0
        // note r is initially set to b a couple lines above
        cublasErrorHandler(cublasDgemv(handle, CUBLAS_OP_N, 3 * N, 3 * N, &neg_one, A,
                    3 * N, x, 1, &one, r, 1),
                __LINE__);

        // Z = M^-1*R
        precondition_z<<<3 * N, 1>>>(3 * N, A, r, z);
        cudaErrorHandler(cudaGetLastError(), __LINE__ - 1);

        // P = Z
        cublasErrorHandler(cublasDcopy(handle, 3 * N, z, 1, p, 1), __LINE__);

        // This line was used for testing cuda cdvdw in hkust to prevent some non-NAN polarization values
        //system->polar_max_iter = 0;
        
        for (iterations = 0; iterations < system->polar_max_iter; iterations++) {
            // alpha = R^tZ/P^tAP
            cublasErrorHandler(cublasDdot(handle, 3 * N, r, 1, z, 1, &alpha), __LINE__);
            cublasErrorHandler(cublasDgemv(handle, CUBLAS_OP_N, 3 * N, 3 * N, &one, A,
                        3 * N, p, 1, &zero, tmp, 1),
                    __LINE__);
            cublasErrorHandler(cublasDdot(handle, 3 * N, p, 1, tmp, 1, &result),
                    __LINE__);
            alpha /= result;

            // X = X + alpha*P
            cublasErrorHandler(cublasDaxpy(handle, 3 * N, &alpha, p, 1, x, 1),
                    __LINE__);

            // save old R, D
            cublasErrorHandler(cublasDcopy(handle, 3 * N, r, 1, r_prev, 1), __LINE__);
            cublasErrorHandler(cublasDcopy(handle, 3 * N, z, 1, z_prev, 1), __LINE__);

            // R = R - alpha*AP
            alpha *= -1;
            cublasErrorHandler(cublasDaxpy(handle, 3 * N, &alpha, tmp, 1, r, 1),
                    __LINE__);

            // Z = M^-1*R
            precondition_z<<<3 * N, 1>>>(3 * N, A, r, z);
            cudaErrorHandler(cudaGetLastError(), __LINE__ - 1);

            // beta = Z^tR/Z_prev^tR_prev
            cublasErrorHandler(cublasDdot(handle, 3 * N, z, 1, r, 1, &beta), __LINE__);
            cublasErrorHandler(cublasDdot(handle, 3 * N, z_prev, 1, r_prev, 1, &result),
                    __LINE__);
            beta /= result;

            // P = Z + beta*P
            cublasErrorHandler(cublasDcopy(handle, 3 * N, z, 1, tmp, 1), __LINE__);
            cublasErrorHandler(cublasDaxpy(handle, 3 * N, &beta, p, 1, tmp, 1),
                    __LINE__);
            cublasErrorHandler(cublasDcopy(handle, 3 * N, tmp, 1, p, 1), __LINE__);
        }

        cudaErrorHandler(
                cudaMemcpy(host_x, x, 3 * N * sizeof(double), cudaMemcpyDeviceToHost),
                __LINE__);

        potential = 0.0;
        for (molecule_ptr = system->molecules, i = 0; molecule_ptr;
                molecule_ptr = molecule_ptr->next) {
            for (atom_ptr = molecule_ptr->atoms; atom_ptr;
                    atom_ptr = atom_ptr->next, i++) {

                atom_ptr->mu[0] = (double)host_x[3 * i];
                atom_ptr->mu[1] = (double)host_x[3 * i + 1];
                atom_ptr->mu[2] = (double)host_x[3 * i + 2];

                potential += atom_ptr->mu[0] *
                    (atom_ptr->ef_static[0] + atom_ptr->ef_static_self[0]);
                potential += atom_ptr->mu[1] *
                    (atom_ptr->ef_static[1] + atom_ptr->ef_static_self[1]);
                potential += atom_ptr->mu[2] *
                    (atom_ptr->ef_static[2] + atom_ptr->ef_static_self[2]);
            }
        }

        potential *= -0.5;

        free(host_x);
        free(host_b);
        free(host_basis);
        free(host_recip_basis);
        free(host_pos);
        free(host_pols);
        cudaErrorHandler(cudaFree(x), __LINE__);
        cudaErrorHandler(cudaFree(A), __LINE__);
        cudaErrorHandler(cudaFree(r), __LINE__);
        cudaErrorHandler(cudaFree(z), __LINE__);
        cudaErrorHandler(cudaFree(p), __LINE__);
        cudaErrorHandler(cudaFree(tmp), __LINE__);
        cudaErrorHandler(cudaFree(r_prev), __LINE__);
        cudaErrorHandler(cudaFree(z_prev), __LINE__);
        cudaErrorHandler(cudaFree(pos), __LINE__);
        cudaErrorHandler(cudaFree(pols), __LINE__);
        cublasErrorHandler(cublasDestroy(handle), __LINE__);

        system->observables->polarization_energy = (double)potential;
        return NULL;
    }
}
