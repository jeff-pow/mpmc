#include <mc.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#define TWOoverHBAR 2.6184101e11    //K^-1 s^-1
#define cHBAR 7.63822291e-12        //Ks //HBAR is already taken to be in Js
#define halfHBAR 3.81911146e-12     //Ks
#define au2invsec 4.13412763705e16  //s^-1 a.u.^-1
#define FINITE_DIFF 0.01            //too small -> vdw calc noises becomes a problem
#define STOCHASTIC_ITERS 50
#define LANCZOS_SIZE 50


//only run dsyev from LAPACK if compiled with -llapack, otherwise report an error
#ifdef VDW
//prototype for dsyev (LAPACK)
extern void dsyev_(char *, char *, int *, double *, int *, double *, double *, int *, int *);
extern void dstev_(char *, int *, double *, double *, double *, int *, double *, int *);
#else
void dsyev_(char *a, char *b, int *c, double *d, int *e, double *f, double *g, int *h, int *i) {
    error(
        "ERROR: Not compiled with Linear Algebra VDW.\n");
    die(-1);
}
#endif

void print_mtx(double *matrix, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%14.10f ", matrix[i * dim + j]);
        }
        printf("\n");
    }
}

/**
* Calculates the two-norm of a vector
*/
static double norm(double *vec, int dim) {
    double n = 0;
    for (int j = 0; j < dim; j++) {
        n += vec[j] * vec[j];
    }
    n = sqrt(n);
    return n;
}

static void normalize(double *vec, int dim) {
    double n = norm(vec, dim);
    for (int j = 0; j < dim; j++) {
        vec[j] /= n;
    }
}

static void mtx_vec_mult(double *mtx, double *vec, double *result, int dim) {
    for (int i = 0; i < dim; i++) {
        result[i] = 0;
        for (int j = 0; j < dim; j++) {
            result[i] += vec[j] * mtx[i * dim + j];
        }
    }
}

static void vec_vec_mult(double *v1, double *v2, int dim, double *result) {
    for (int i = 0; i < dim; i++) {
        result[i] = v1[i] * v2[i];
    }
}

static double dot(double *v1, double *v2, int dim) {
    double result = 0;
    for (int i = 0; i < dim; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

static void reorthoganlize(double *vec, int dim, double *vs, int vs_len) {
    for (int i = 0; i < vs_len; i++) {
        for (int j = 0; j < dim; j++) {
            double d = dot(&vs[i * dim], &vs[i * dim], dim);
            vec[j] -= d * vs[i * dim];
        }
        normalize(vec, dim);
    }
}

static void lanczos(double *matrix, double *v, int m, int dim, double *alphas, double *betas, bool do_reorthoganlization) {
    // BUG: Does this need to be normed again?
    normalize(v, dim);

    double *w = calloc(dim, sizeof(double));
    mtx_vec_mult(matrix, v, w, dim);

    double alpha = dot(w, v, dim);
    for (int i = 0; i < dim; i++) {
        w[i] = w[i] - alpha * v[i];
    }

    double *v_last = calloc(dim, sizeof(double));
    for (int i = 0; i < dim; i++) {
        v_last[i] = v[i];
    }
    alphas[0] = alpha;

    double *vs = calloc(m * dim, sizeof(double));
    for (int i = 0; i < dim; i++) {
        vs[i] = v[i];
    }

    for (int i = 0; i < m - 1; i++) {
        double beta = norm(w, dim);
        if (beta == 0) {
            for (int j = 0; j < dim; j++) {
                double r = (double)rand() / RAND_MAX;
                if (r < .5) {
                    v[j] = -1;
                }
                else {
                    v[j] = 1;
                }
            }
            normalize(v, dim);
            // reorthoganlize(v, dim, vs, i + 1);
        }
        else {
            for (int j = 0; j < dim; j++) {
                v[j] = w[j] / beta;
            }
        }
        if (do_reorthoganlization) {
            // reorthoganlize(v, dim, vs, i + 1);
        }
        for (int j = 0; j < dim; j++) {
            w[j] = 0;
            for (int k = 0; k < dim; k++) {
                w[j] += matrix[j * dim + k] * v[j];
            }
        }

        mtx_vec_mult(matrix, v, w, dim);

        alpha = dot(w, v, dim);
        for (int j = 0; j < dim; j++) {
            w[j] = w[j] - alpha * v[j] - beta * v_last[j];
        }
        for (int j = 0; j < dim; j++) {
            v_last[j] = v[j];
        }
        alphas[i + 1] = alpha;
        betas[i] = beta;
        for (int j = 0; j < dim; j++) {
            vs[i * dim + j] = v[j];
        }
    }
    free(w);
    free(v_last);
    free(vs);
}

static double slq_lanczos(double *matrix, int num_iters, int dim, int lanczos_size) {
    double sum = 0;
    srand(time(0));
    for (int i = 0; i < num_iters; i++) {
        double *rademacher = calloc(dim, sizeof(double));
        for (int j = 0; j < dim; j++) {
            double r = (double)rand() / RAND_MAX;
            if (r < .5) {
                rademacher[j] = -1;
            }
            else {
                rademacher[j] = 1;
            }
        }
        normalize(rademacher, dim);

        double *diag = calloc(lanczos_size, sizeof(double));
        double *sub_diag = calloc((lanczos_size - 1), sizeof(double));
        lanczos(matrix, rademacher, lanczos_size, dim, diag, sub_diag, true);

        double *work = calloc((3 * lanczos_size - 2), sizeof(double));
        double *eigvecs = calloc(lanczos_size * lanczos_size, sizeof(double));
        int info = 0;
        char job = 'V';
        // Eigvecs are placed in eigvecs, eigvals are placed in diag
        dstev_(&job, &lanczos_size, diag, sub_diag, eigvecs, &lanczos_size, work, &info);

        for (int j = 0; j < lanczos_size; j++) {
            sum += sqrt(diag[j]) * eigvecs[j] * eigvecs[j];
        }
        free(diag);
        free(sub_diag);
        free(work);
        free(eigvecs);
        free(rademacher);
    }
    return sum * dim / num_iters;
}

static double *build_C(int A_dim, int C_dim, int offset, system_t *system) {
    double *C = calloc(A_dim * A_dim, sizeof(double));
    for (int i = 0; i < A_dim; i++) {
        for (int j = 0; j < A_dim; j++) {
            if (j < offset || j >= C_dim + offset || i < offset || i >= C_dim + offset) {
                C[i * A_dim + j] = 0;
            }
            else {
                C[i * A_dim + j] = system->A_matrix[i][j] * system->atom_array[i / 3]->omega * system->atom_array[j / 3]->omega
                    * sqrt(system->atom_array[i / 3]->polarizability * system->atom_array[j / 3]->polarizability);
            }
        }
    }
    return C;
}

static double calc_e_iso(system_t *system, molecule_t *mptr) {
    int nstart, nsize;   // , curr_dimM;  (unused variable)
    double e_iso;        //total vdw energy of isolated molecules
    double *Cm_iso;  //matrix Cm_isolated
    molecule_t *molecule_ptr;
    atom_t *atom_ptr;
    int A_dim = 3 * system->natoms;

    nstart = nsize = 0;  //loop through each individual molecule
    for (molecule_ptr = system->molecules; molecule_ptr; molecule_ptr = molecule_ptr->next) {
        if (molecule_ptr != mptr) {  //count atoms then skip to next molecule
            for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) nstart++;
            continue;
        }

        //now that we've found the molecule of interest, count natoms, and calc energy
        for (atom_ptr = molecule_ptr->atoms; atom_ptr; atom_ptr = atom_ptr->next) nsize++;

        //build matrix for calculation of vdw energy of isolated molecule
        int C_dim = 3 * nsize;
        int offset = 3 * nstart;
        Cm_iso = build_C(A_dim, C_dim, offset, system);
        //diagonalize M and extract eigenvales -> calculate energy
        //eigvals = lapack_diag(Cm_iso, 1);  //no eigenvectors
        e_iso = slq_lanczos(Cm_iso, STOCHASTIC_ITERS, 3 * system->natoms, LANCZOS_SIZE);

        //convert a.u. -> s^-1 -> K
        return e_iso * au2invsec * halfHBAR;
    }

    //unmatched molecule
    return NAN;  //we should never get here
}

static double sum_eiso_vdw(system_t *system) {
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
                system->vdw_eiso_info = calloc(1, sizeof(vdw_t));  //allocate space
                vpscan = system->vdw_eiso_info;  //set scan pointer
            } else {
                for (vpscan = system->vdw_eiso_info; vpscan->next != NULL; vpscan = vpscan->next);
                vpscan->next = calloc(1, sizeof(vdw_t));  //allocate space
                vpscan = vpscan->next;
            }  //done scanning and malloc'ing

            //set values
            strncpy(vpscan->mtype, mp->moleculetype, MAXLINE);  //assign moleculetype
            vpscan->energy = calc_e_iso(system, mp);  //assign energy
            if (isfinite(vpscan->energy) == 0) {                //if nan, then calc_e_iso failed
                sprintf(linebuf, "VDW: Problem in calc_e_iso.\n");
                output(linebuf);
                die(-1);
            }
            //otherwise count the energy and move to the next molecule
            e_iso += vpscan->energy;

        }  //vp==NULL
    }      //mp loop

    ////all of this logic is actually really bad if we're doing surface fitting, since omega will change... :-(
    //free everything so we can recalc next step
    if (system->ensemble == ENSEMBLE_SURF_FIT) {
        free_vdw_eiso(system->vdw_eiso_info);
        system->vdw_eiso_info = NULL;
    }

    return e_iso;
}

//returns interaction VDW energy
double fast_vdw(system_t *system) {
    int N;                           //  dimC;  (unused variable)  //number of atoms, number of non-zero rows in C-Matrix
    double e_total, e_iso;           //total energy, isolation energy (atoms @ infinity)

    N = system->natoms;

    clock_t start_time = clock();
    //calculate energy vdw of isolated molecules
    e_iso = sum_eiso_vdw(system);
    printf("e iso elapsed: %f\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);
    printf("Fast e_iso: %14.10e\n", e_iso);
    printf("done eiso\n");

    //Build the C_Matrix
    double *Cm = build_C(3 * N, 3 * N, 0, system);

    e_total = slq_lanczos(Cm, STOCHASTIC_ITERS, 3 * N, LANCZOS_SIZE);
    e_total *= au2invsec * halfHBAR;  //convert a.u. -> s^-1 -> K
    printf("e_total elapsed: %f\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);

    //cleanup and return
    free(Cm);

    printf("Fast e_total: %14.10e\n", e_total);
    printf("Fast energy: %14.10e\n", e_total - e_iso);
    printf("done all\n");
    printf("\n\n");

    return e_total - e_iso;
}
