/**
 * @author Jeff Powell (2023)
 */

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <cuda.h>
#include <defines.h>

// This resulted in the same times across 64, 128, 256, and 512. I just went with a middle ground...
#define THREADS 128
#define MAXFVALUE 1.0e13f
#define halfHBAR 3.81911146e-12     //Ks
#define cHBAR 7.63822291e-12        //Ks //HBAR is already taken to be in Js
#define FINITE_DIFF 0.01            //too small -> vdw calc noises becomes a problem
#define TWOoverHBAR 2.6184101e11    //K^-1 s^-1

__constant__ double basis[9];
__constant__ double recip_basis[9];

__global__ static void build_c(int N, double *A, double *omegas, double *pols, double *C, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim * dim) return;
    int row = i / dim;
    int col = i % dim;
    C[col * dim + row] = A[row * dim + col] * pols[row] * pols[col] * 
        sqrt(omegas[row] * omegas[col]);
}

__global__
static void print_basis_sets() {
    for (int i = 0; i < 3 * 3; i++) {
        printf("%8.5f ", basis[i]);
        if ((i + 1) % 3 == 0 && i != 0) {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 0; i < 3 * 3; i++) {
        printf("%8.5f ", recip_basis[i]);
        if ((i + 1) % 3 == 0 && i != 0) {
            printf("\n");
        }
    }
    printf("\n");
}

__global__ static void print_a(int N, double *A) {
    for (int i = 0; i < 3 * 3 * N * N; i++) {
        printf("%8.5f ", A[i]);
        if ((i + 1) % (N * 3) == 0 && i != 0) {
            printf("\n");
        }
    }
    printf("\n");
}

/**
 * Method uses exponential polarization regardless of method requested in input
 * script
 */
__global__ static void build_a(int N, double *A, const double damp, double3 *pos, double *pols, int damp_type) {
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
                case DAMPING_EXPONENTIAL:
                    // damping terms
                    expr = exp(-damp * r);
                    damping_term1 = 1.0f - expr * (0.5f * damp2 * r2 + damp * r + 1.0f);
                    damping_term2 = 1.0f - expr * (damp3 * r * r2 / 6.0f + 0.5f * damp2 * r2 +
                        damp * r + 1.0f);

                    // construct the Tij tensor field, unrolled by hand to avoid conditional
                    // on the diagonal terms
                    damping_term1 *= r3;
                    damping_term2 *= -3.0f * r5;
                    break;
                case DAMPING_EXPONENTIAL_FIXED: {
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
                default:
                    printf("Damping type has not been implemented for many body van der waals.\n");
                    printf("Error in vdw.cu\n");
            }


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

__global__
void build_c_matrix(int matrix_size, int dim, double *A, double *pols, double *omegas, double *device_C_matrix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= matrix_size) return;

    int row = i % dim;
    int col = i / dim;
    device_C_matrix[i] = A[i] * omegas[col / 3] * omegas[row / 3] * 
        sqrt(pols[col / 3] * pols[row / 3]);
}

__global__
void build_c_matrix_with_offset(int C_dim, int A_dim, int offset, double *A, double *pols, double *omegas, double *C_matrix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_dim * A_dim) return;
    int row = i % A_dim;
    int col = i / A_dim;
    if (col < offset || col >= C_dim + offset || row < offset || row >= C_dim + offset) {
        C_matrix[i] = 0;
    }
    else {
        double a1 = pols[row / 3]; // alphas
        double a2 = pols[col / 3];
        double w1 = omegas[row / 3]; // omegas
        double w2 = omegas[col / 3];
        C_matrix[i] = A[i] * w1 * w2 * sqrt(a1 * a2);
    }
}

__global__
void print_arr(int arr_size, double *arr) {
    for (int i = 0; i < arr_size; i++) {
        printf("%.3le\n", arr[i]);
    }
}

__global__
void build_kinvsqrt(int matrix_size, int dim, double *pols, double *omegas, double *device_invKsqrt_matrix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= matrix_size) return;

    int row = i % dim;
    int col = i / dim;
    if (row != col) {
        return;
    }
    device_invKsqrt_matrix[row * 3 * dim / 3 + col] = sqrt((pols[row / 3] * omegas[row / 3] * omegas[row / 3]));
}

__global__ static void print_matrix(int dim, double *A) {
    for (int i = 0; i < dim * dim; i++) {
        printf("%.3le ", A[i]);
        if ((i + 1) % (dim) == 0 && i != 0) {
            printf("\n");
        }
    }
    printf("\n");
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

static double eigen2energy(double *eigvals, int dim, double temperature) {
    int i;
    double rval = 0;

    if (eigvals == NULL) return 0;

    for (i = 0; i < dim; i++) {
        if (eigvals[i] < 0) eigvals[i] = 0;
        //		rval += wtanh(sqrt(eigvals[i]), temperature);
        rval += sqrt(eigvals[i]);
        //printf("eigs[%d]: %le\n", i, eigvals[i]);
    }
    return rval;
}


//calculate energies for isolated molecules
//if we don't know it, calculate it and save the value
// This seems like it might be ripe for speeding up, but I checked the runtime and it was extremely minimal. The fact
// that I didn't use an offset hardly makes a difference from my testing.
static double calc_e_iso(system_t *system, molecule_t *mptr, double *device_A_matrix, int A_dim, double *device_pols,
                         double *device_omegas) {
    int nstart, nsize;   // , curr_dimM;  (unused variable)
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;

    nstart = nsize = 0;  //loop through each individual molecule
    for (molecule_ptr = system->molecules; molecule_ptr; molecule_ptr = molecule_ptr->next) {
        if (molecule_ptr != mptr) {  //count atoms then skip to next molecule
            for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) nstart++;
            continue;
        }

        //now that we've found the molecule of interest, count natoms, and calc energy
        for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) nsize++;

        //build matrix for calculation of vdw energy of isolated molecule
        double *device_C_matrix;
        int dim = 3 * nsize;
        int offset = 3 * nstart;
        int matrix_size = A_dim * A_dim;
        cudaErrorHandler(cudaMalloc((void **) &device_C_matrix, matrix_size * sizeof(double)), __LINE__);
        int blocks = (matrix_size + THREADS - 1) / THREADS;
        build_c_matrix_with_offset<<<blocks, THREADS>>>(dim, A_dim, offset, device_A_matrix, device_pols, device_omegas, device_C_matrix);
        cudaDeviceSynchronize();
        cudaErrorHandler(cudaGetLastError(), __LINE__ - 1);
        //diagonalize M and extract eigenvales -> calculate energy

        int *devInfo;
        double *d_work;
        double *d_W;
        int lwork = 0;
        cudaErrorHandler(cudaMalloc((void **)&d_W, A_dim * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&d_work, A_dim * sizeof(double)), __LINE__);
        cudaErrorHandler(cudaMalloc((void **)&devInfo, sizeof(int)), __LINE__);
        cudaDeviceSynchronize();
        cusolverDnHandle_t cusolverH;
        cusolverDnCreate(&cusolverH);
        cudaDeviceSynchronize();

        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        // Find optimal workspace size
        cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, A_dim, device_C_matrix, A_dim, d_W, &lwork);
        cudaDeviceSynchronize();
        cudaErrorHandler(cudaFree(d_work), __LINE__);
        cudaDeviceSynchronize();
        cudaErrorHandler(cudaMalloc( (void **) &d_work, lwork * sizeof(double)), __LINE__);
        cudaDeviceSynchronize();
        // Solve for eigenvalues
        cusolverDnDsyevd(cusolverH, jobz, uplo, A_dim, device_C_matrix, A_dim, d_W, d_work, lwork, devInfo);
        cudaDeviceSynchronize();

        double *host_eigenvalues = (double *)malloc(A_dim * sizeof(double));
        cudaErrorHandler(cudaMemcpy(host_eigenvalues, d_W, A_dim * sizeof(double), cudaMemcpyDeviceToHost), __LINE__);
        double e_iso = eigen2energy(host_eigenvalues, A_dim, system->temperature);

        //free memory
        free(host_eigenvalues);
        cusolverDnDestroy(cusolverH);
        cudaErrorHandler(cudaFree(device_C_matrix), __LINE__);
        cudaErrorHandler(cudaFree(d_W), __LINE__);
        cudaErrorHandler(cudaFree(d_work), __LINE__);
        cudaErrorHandler(cudaFree(devInfo), __LINE__);

        //convert a.u. -> s^-1 -> K
        return e_iso * au2invseconds * halfHBAR;
    }

    //unmatched molecule
    return NAN;  //we should never get here
}


static double sum_eiso_vdw(system_t *system, double *device_A_matrix, int A_dim, double *device_pols, double *device_omegas) {
    char linebuf[MAXLINE];
    double e_iso = 0;
    molecule_t *mp;
    // atom_t * ap;  (unused variable)
    vdw_t *vp;
    vdw_t *vpscan;

    //loop through molecules. if not known, calculate, store and count. otherwise just count.
    for (mp = system->molecules; mp; mp = mp->next) {
        for (vp = system->vdw_eiso_info; vp != NULL; vp = vp->next) {  //loop through all vp's
            if (strncmp(vp->mtype, mp->moleculetype, MAXLINE) == 0) {
                e_iso += vp->energy;  //count energy
                break;                //break out of vp loop. the current molecule is accounted for now. go to the next molecule
            } else
                continue;  //not a match, check the next vp
        }                  //vp loop

        if (vp == NULL) {  //if the molecule was unmatched, we need to grow the list
            // end of vp list and we haven't matched yet -> grow vdw_eiso_info
            // scan to the last non-NULL element
            if (system->vdw_eiso_info == NULL) {
                system->vdw_eiso_info = (vdw_t *)calloc(1, sizeof(vdw_t));  //allocate space
                vpscan = system->vdw_eiso_info;  //set scan pointer
            } else {
                for (vpscan = system->vdw_eiso_info; vpscan->next != NULL; vpscan = vpscan->next);
                vpscan->next = (vdw_t *)calloc(1, sizeof(vdw_t));  //allocate space
                vpscan = vpscan->next;
            }  //done scanning and malloc'ing

            //set values
            strncpy(vpscan->mtype, mp->moleculetype, MAXLINE);  //assign moleculetype
            vpscan->energy = calc_e_iso(system, mp, device_A_matrix, A_dim, device_pols, device_omegas);  //assign energy
            if (isfinite(vpscan->energy) == 0) {                //if nan, then calc_e_iso failed
                sprintf(linebuf, "VDW: Problem in calc_e_iso.\n");
                exit(1);
            }
            //otherwise count the energy and move to the next molecule
            e_iso += vpscan->energy;

        }  //vp==NULL
    }      //mp loop

    ////all of this logic is actually really bad if we're doing surface fitting, since omega will change... :-(
    //free everything so we can recalc next step
    if (system->ensemble == ENSEMBLE_SURF_FIT) {
        system->vdw_eiso_info = NULL;
    }

    return e_iso;
}

//calculate T matrix element for a particular separation
static double e2body(system_t *system, atom_t *atom, pair_t *pair, double r) {
    double energy = 0;
    double lr = system->polar_damp * r;
    double lr2 = lr * lr;
    double lr3 = lr * lr2;
    double Txx = pow(r, -3) * (-2.0 + (0.5 * lr3 + lr2 + 2 * lr + 2) * exp(-lr));
    double Tyy = pow(r, -3) * (1 - (0.5 * lr2 + lr + 1) * exp(-lr));
    double *eigvals = (double *) malloc(36 * sizeof(double));
    double *T_matrix = (double *) malloc(36 * sizeof(double));

    //only the sub-diagonals are non-zero
    T_matrix[1] = T_matrix[2] = T_matrix[4] = T_matrix[5] = T_matrix[6] = T_matrix[8] = T_matrix[9] = T_matrix[11] = 0;
    T_matrix[12] = T_matrix[13] = T_matrix[15] = T_matrix[16] = T_matrix[19] = T_matrix[20] = T_matrix[22] = T_matrix[23] = 0;
    T_matrix[24] = T_matrix[26] = T_matrix[27] = T_matrix[29] = T_matrix[30] = T_matrix[31] = T_matrix[33] = T_matrix[34] = 0;

    //true diagonals
    T_matrix[0] = T_matrix[7] = T_matrix[14] = (atom->omega) * (atom->omega);
    T_matrix[21] = T_matrix[28] = T_matrix[35] = (pair->atom->omega) * (pair->atom->omega);

    //sub-diagonals
    T_matrix[3] = T_matrix[18] =
        (atom->omega) * (pair->atom->omega) * sqrt(atom->polarizability * pair->atom->polarizability) * Txx;
    T_matrix[10] = T_matrix[17] = T_matrix[25] = T_matrix[32] =
        (atom->omega) * (pair->atom->omega) * sqrt(atom->polarizability * pair->atom->polarizability) * Tyy;

    //eigvals = lapack_diag(M, 1);
    //energy = eigen2energy(eigvals, 6, system->temperature);
    char job = 'N';
    char uplo = 'L';  //operate on lower triagle
    int workSize = -1;
    int rval = 0;
    int dim = 6;
    double *workArr = (double *) malloc(sizeof(double));
    dsyev_(&job, &uplo, &dim, T_matrix, &dim, eigvals, workArr, &workSize, &rval);
    //now optimize work array size is stored as work[0]
    workSize = (int)workArr[0];
    workArr = (double *) realloc(workArr, workSize * sizeof(double));
    //diagonalize
    dsyev_(&job, &uplo, &dim, T_matrix, &dim, eigvals, workArr, &workSize, &rval);

    //subtract energy of atoms at infinity
    //	energy -= 3*wtanh(atom->omega, system->temperature);
    energy -= 3 * atom->omega;
    //	energy -= 3*wtanh(pair->atom->omega, system->temperature);
    energy -= 3 * pair->atom->omega;
    for (int i = 0; i < dim; i++) {
        energy += sqrt(eigvals[i]);
    }

    free(eigvals);
    free(workArr);
    free(T_matrix);

    return energy * au2invseconds * halfHBAR;
}

//with damping
static double twobody(system_t *system) {
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
    pair_t *pair_ptr;
    double energy = 0;

    //for each pair
    for (molecule_ptr = system->molecules; molecule_ptr; molecule_ptr = molecule_ptr->next) {
        for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) {
            for (pair_ptr = atom_ptr->pairs; pair_ptr; pair_ptr = pair_ptr->next) {
                //skip if frozen
                if (pair_ptr->frozen) continue;
                //skip if they belong to the same molecule
                if (molecule_ptr == pair_ptr->molecule) continue;
                //skip if distance is greater than cutoff
                if (pair_ptr->rimg > system->pbc->cutoff) continue;
                //check if fh is non-zero
                if (atom_ptr->polarizability == 0 || pair_ptr->atom->polarizability == 0 ||
                    atom_ptr->omega == 0 || pair_ptr->atom->omega == 0) continue;  //no vdw energy

                //calculate two-body energies
                energy += e2body(system, atom_ptr, pair_ptr, pair_ptr->rimg);
            }
        }
    }

    return energy;
}

// feynman-hibbs using 2BE (shitty)
static double fh_vdw_corr_2be(system_t *system) {
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
    pair_t *pair_ptr;
    double rm;                 //reduced mass
    double w1, w2;             //omegas
    double a1, a2;             //alphas
    double cC;                 //leading coefficient to r^-6
    double dv, d2v, d3v, d4v;  //derivatives
    double corr = 0;           //correction to the energy
    double corr_single;        //single vdw interaction energy

    //for each pair
    for (molecule_ptr = system->molecules; molecule_ptr; molecule_ptr = molecule_ptr->next) {
        for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) {
            for (pair_ptr = atom_ptr->pairs; pair_ptr; pair_ptr = pair_ptr->next) {
                //skip if frozen
                if (pair_ptr->frozen) continue;
                //skip if they belong to the same molecule
                if (molecule_ptr == pair_ptr->molecule) continue;
                //skip if distance is greater than cutoff
                if (pair_ptr->rimg > system->pbc->cutoff) continue;
                //fetch alphas and omegas
                a1 = atom_ptr->polarizability;
                a2 = pair_ptr->atom->polarizability;
                w1 = atom_ptr->omega;
                w2 = pair_ptr->atom->omega;
                if (w1 == 0 || w2 == 0 || a1 == 0 || a2 == 0) continue;  //no vdw energy
                // 3/4 hbar/k_B(Ks) omega(s^-1)  Ang^6
                cC = 1.5 * cHBAR * w1 * w2 / (w1 + w2) * au2invseconds * a1 * a2;
                // reduced mass
                rm = AMU2KG * (molecule_ptr->mass) * (pair_ptr->molecule->mass) /
                    ((molecule_ptr->mass) + (pair_ptr->molecule->mass));

                //derivatives
                dv = 6.0 * cC * pow(pair_ptr->rimg, -7);
                d2v = dv * (-7.0) / pair_ptr->rimg;
                if (system->feynman_hibbs_order >= 4) {
                    d3v = d2v * (-8.0) / pair_ptr->rimg;
                    d4v = d3v * (-9.0) / pair_ptr->rimg;
                }

                //2nd order correction
                corr_single = pow(METER2ANGSTROM, 2) * (HBAR * HBAR / (24.0 * KB * system->temperature * rm)) * (d2v + 2.0 * dv / pair_ptr->rimg);
                //4th order correction
                if (system->feynman_hibbs_order >= 4)
                    corr_single += pow(METER2ANGSTROM, 4) * (pow(HBAR, 4) / (1152.0 * pow(KB * system->temperature * rm, 2))) *
                        (15.0 * dv / pow(pair_ptr->rimg, 3) + 4.0 * d3v / pair_ptr->rimg + d4v);

                corr += corr_single;
            }
        }
    }

    return corr;
}

// feynman-hibbs correction - molecular pair finite differencing method
static double fh_vdw_corr(system_t *system) {
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
    pair_t *pair_ptr;
    double rm;                 //reduced mass
    double E[5];               //energy at five points, used for finite differencing
    double dv, d2v, d3v, d4v;  //derivatives
    double corr = 0;           //correction to the energy
    double corr_single;        //single vdw interaction energy
    double h = FINITE_DIFF;    //small dr used for finite differencing //too small -> vdw calculation noise becomes a problem

    //for each pair
    for (molecule_ptr = system->molecules; molecule_ptr; molecule_ptr = molecule_ptr->next) {
        for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) {
            for (pair_ptr = atom_ptr->pairs; pair_ptr; pair_ptr = pair_ptr->next) {
                //skip if frozen
                if (pair_ptr->frozen) continue;
                //skip if they belong to the same molecule
                if (molecule_ptr == pair_ptr->molecule) continue;
                //skip if distance is greater than cutoff
                if (pair_ptr->rimg > system->pbc->cutoff) continue;
                //check if fh is non-zero
                if (atom_ptr->polarizability == 0 || pair_ptr->atom->polarizability == 0 ||
                    atom_ptr->omega == 0 || pair_ptr->atom->omega == 0) continue;  //no vdw energy

                //calculate two-body energies
                E[0] = e2body(system, atom_ptr, pair_ptr, pair_ptr->rimg - h - h);  //smaller r
                E[1] = e2body(system, atom_ptr, pair_ptr, pair_ptr->rimg - h);
                E[2] = e2body(system, atom_ptr, pair_ptr, pair_ptr->rimg);      //current r
                E[3] = e2body(system, atom_ptr, pair_ptr, pair_ptr->rimg + h);  //larger r
                E[4] = e2body(system, atom_ptr, pair_ptr, pair_ptr->rimg + h + h);

                //derivatives (Numerical Methods Using Matlab 4E 2004 Mathews/Fink 6.2)
                dv = (E[3] - E[1]) / (2.0 * h);
                d2v = (E[3] - 2.0 * E[2] + E[1]) / (h * h);
                d3v = (E[4] - 2 * E[3] + 2 * E[1] - E[0]) / (2 * pow(h, 3));
                d4v = (E[4] - 4 * E[3] + 6 * E[2] - 4 * E[1] + E[0]) / pow(h, 4);

                // reduced mass
                rm = AMU2KG * (molecule_ptr->mass) * (pair_ptr->molecule->mass) /
                    ((molecule_ptr->mass) + (pair_ptr->molecule->mass));

                //2nd order correction
                corr_single = pow(METER2ANGSTROM, 2) * (HBAR * HBAR / (24.0 * KB * system->temperature * rm)) * (d2v + 2.0 * dv / pair_ptr->rimg);
                //4th order correction
                if (system->feynman_hibbs_order >= 4)
                    corr_single += pow(METER2ANGSTROM, 4) * (pow(HBAR, 4) / (1152.0 * pow(KB * system->temperature * rm, 2))) *
                        (15.0 * dv / pow(pair_ptr->rimg, 3) + 4.0 * d3v / pair_ptr->rimg + d4v);

                corr += corr_single;
            }
        }
    }

    return corr;
}

// long-range correction
static double lr_vdw_corr(system_t *system) {
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
    pair_t *pair_ptr;
    double w1, w2;    //omegas
    double a1, a2;    //alphas
    double cC;        //leading coefficient to r^-6
    double corr = 0;  //correction to the energy

    //skip if PBC isn't set-up
    if (system->pbc->volume == 0) {
        fprintf(stderr, "VDW: PBC not set-up. Did you define your basis? Skipping LRC.\n");
        return 0;
    }

    for (molecule_ptr = system->molecules; molecule_ptr; molecule_ptr = molecule_ptr->next) {
        for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) {
            for (pair_ptr = atom_ptr->pairs; pair_ptr; pair_ptr = pair_ptr->next) {
                //skip if frozen
                if (pair_ptr->frozen) continue;
                //skip if same molecule  // don't do this... this DOES contribute to LRC
                //					if ( molecule_ptr == pair_ptr->molecule ) continue;
                //fetch alphas and omegas
                a1 = atom_ptr->polarizability;
                a2 = pair_ptr->atom->polarizability;
                w1 = atom_ptr->omega;
                w2 = pair_ptr->atom->omega;
                if (w1 == 0 || w2 == 0 || a1 == 0 || a2 == 0) continue;  //no vdw energy
                // 3/4 hbar/k_B(Ks) omega(s^-1)  Ang^6
                cC = 1.5 * cHBAR * w1 * w2 / (w1 + w2) * au2invseconds * a1 * a2;

                // long-range correction
                corr += -4.0 / 3.0 * M_PI * cC * pow(system->pbc->cutoff, -3) / system->pbc->volume;
            }
        }
    }

    return corr;
}

void *vdw_cuda(void *systemptr) {
    system_t *system = (system_t *)systemptr;
    int N = system->natoms;
    int matrix_size = 3 * 3 * N * N;
    int dim = 3 * N;

    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
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

    for (i = 0; i < N; i++) {
        host_omegas[i] = system->atom_array[i]->omega;
    }
    cudaErrorHandler(cudaMemcpy(device_omegas, host_omegas, N * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
    cudaDeviceSynchronize();

    int blocks = (matrix_size + THREADS - 1) / THREADS;
    build_c_matrix<<<blocks, THREADS>>>(matrix_size, dim, device_A_matrix, device_pols, device_omegas, device_C_matrix);
    cudaErrorHandler(cudaGetLastError(), __LINE__ - 1);
    cudaDeviceSynchronize();

    int *devInfo;
    double *d_work;
    double *d_W;
    int lwork = 0;
    cudaErrorHandler(cudaMalloc((void **)&d_W, dim * sizeof(double)), __LINE__);
    cudaErrorHandler(cudaMalloc((void **)&d_work, dim * sizeof(double)), __LINE__);
    cudaErrorHandler(cudaMalloc((void **)&devInfo, sizeof(int)), __LINE__);
    cudaDeviceSynchronize();
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    cudaDeviceSynchronize();

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // Find optimal workspace size
    cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, dim, device_C_matrix, dim, d_W, &lwork);
    cudaErrorHandler(cudaFree(d_work), __LINE__);
    cudaDeviceSynchronize();
    cudaErrorHandler(cudaMalloc( (void **) &d_work, lwork * sizeof(double)), __LINE__);
    cudaDeviceSynchronize();
    // Solve for eigenvalues
    cusolverDnDsyevd(cusolverH, jobz, uplo, dim, device_C_matrix, dim, d_W, d_work, lwork, devInfo);
    cudaDeviceSynchronize();

    double *host_eigenvalues = (double *)malloc(dim * sizeof(double));
    cudaErrorHandler(cudaMemcpy(host_eigenvalues, d_W, dim * sizeof(double), cudaMemcpyDeviceToHost), __LINE__);
    cudaDeviceSynchronize();
    double e_total = 0;
    e_total = eigen2energy(host_eigenvalues, dim, system->temperature);
    e_total *= au2invseconds * halfHBAR;

    double e_iso = sum_eiso_vdw(system, device_A_matrix, dim, device_pols, device_omegas);

    //vdw energy comparison
    if (system->polarvdw == 3) {
        printf("VDW Two-Body | Many Body = %lf | %lf\n", twobody(system), e_total - e_iso);
    }

    double fh_corr, lr_corr;
    if (system->feynman_hibbs) {
        if (system->vdw_fh_2be)
            fh_corr = fh_vdw_corr_2be(system);  //2be method
        else
            fh_corr = fh_vdw_corr(system);  //mpfd
    } else
        fh_corr = 0;

    if (system->rd_lrc)
        lr_corr = lr_vdw_corr(system);
    else
        lr_corr = 0;


    free(host_omegas);
    free(host_eigenvalues);
    free(host_basis);
    free(host_recip_basis);
    free(host_pos);
    free(host_pols);
    cudaErrorHandler(cudaFree(device_C_matrix), __LINE__);
    cudaErrorHandler(cudaFree(device_A_matrix), __LINE__);
    cudaErrorHandler(cudaFree(device_omegas), __LINE__);
    cudaErrorHandler(cudaFree(device_pols), __LINE__);
    cudaErrorHandler(cudaFree(device_pos), __LINE__);
    cudaErrorHandler(cudaFree(d_W), __LINE__);
    cudaErrorHandler(cudaFree(d_work), __LINE__);
    cudaErrorHandler(cudaFree(devInfo), __LINE__);
    cusolverDnDestroy(cusolverH);
    cudaDeviceSynchronize();



    double energy = e_total - e_iso + fh_corr + lr_corr;
    system->observables->vdw_energy = energy;
    return NULL;
}
}

