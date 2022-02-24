#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "mkl.h"
#include "mkl_vsl.h"
#include "mkl_spblas.h"
#include "omp.h"

// NOTE: all print_matrix functions assume the matrix is in column-major order.
//		 if the matrix is in row-major order, then swap the row and column dimensions
//	     so its transpose is printed instead.

int print_matrix(char *label, int n, int m, double *a) {
    int i, j;
    double val;
    printf("\n%s\n", label);
    for (i = 0; i < n; ++i) {
        printf("\t");
        for (j = 0; j < m - 1; ++j) {
            val = a[i + n * j];
            if (val < 0) {
                printf("  %2.4f,", val);
            } else {
                printf("   %2.4f,", val);
            }
        }
        // j = m - 1
        val = a[i + n * j];
        if (val < 0) {
            printf("  %2.4f;...", val);
        } else {
            printf("   %2.4f;...", val);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int print_matrix(char *label, int n, int m, int *a) {
    int i, j;
    int val;
    printf("\n%s\n", label);
    for (i = 0; i < n; ++i) {
        printf("\t");
        for (j = 0; j < m - 1; ++j) {
            val = a[i + n * j];
            if (val < 0) {
                printf("  %i,", val);
            } else {
                printf("   %i,", val);
            }
        }
        // j = m - 1
        val = a[i + n * j];
        if (val < 0) {
            printf("  %i;...", val);
        } else {
            printf("   %i;...", val);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int print_matrix(char *label, int n, int m, MKL_INT *a) {
    int i, j;
    MKL_INT val;
    printf("\n%s\n", label);
    for (i = 0; i < n; ++i) {
        printf("\t");
        for (j = 0; j < m - 1; ++j) {
            val = a[i + n * j];
            if (val < 0) {
                printf("  %i,", val);
            } else {
                printf("   %i,", val);
            }
        }
        // j = m - 1
        val = a[i + n * j];
        if (val < 0) {
            printf("  %i;...", val);
        } else {
            printf("   %i;...", val);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

void random_vector(int n, double *v, int seed) {
	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, seed);
	vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n, v, 0, 1);
	vslDeleteStream( &stream );
	return;
}

void random_pos_def_matrix(int n, double *a, int seed) {
	double *rand_mat = (double *) calloc(n * (n+1), sizeof(double));
	random_vector(n * (n + 1), rand_mat, seed);
	// better to use syrk.
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
				n, n, n + 1,
				1.0, rand_mat, n, rand_mat, n,
				0.0, a, n);
	free(rand_mat);
	return;
}

// sparse matrix times dense matrix gives dense matrix
// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/inspector-executor-sparse-blas-execution-routines/mkl-sparse-mm.html

void gen_sjlt(int d, int m, int k, int *rows, int *cols, int*vals, int seed) {
	// The fact that this uses MKL is incidental. Any library for random number generation would suffice.
	int nnz = m * k;  // number of columns "m" times nnz per column "k"
	int i, j;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < k; ++j) {
			cols[j + i * k] = i;
		}
	}
	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, seed);
	for (i = 0; i < m; ++i) {
		// sample with replacement. Sampling without replacement is a pain.
		viRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream,
					 k, // number of samples
					 &rows[i*k], // place to write sample
				 	 0, // lower bound (inclusive)
				  	 d // upper bound (exclusive)
		);
	}
	viRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream, nnz, vals, 0, 2);
	for (i = 0; i < nnz; ++i) {
		vals[i] = 2*vals[i] - 1;
	}
	vslDeleteStream(&stream);
}

// assume a is symmetric and m-by-m; only access n rows or columns.
void apply_sjlt_mkl(int d, int m, int n, int nnz, int *rows, int *cols, int *vals, double *a, double *a_hat, bool c_major) {
	// The point of this function is to specifically call MKL (and only MKL) for the sparse matrix multiplication.
	//

	// Build the matrix in MKL's coo format.
	double *dvals = (double*) calloc(nnz, sizeof(double));
    MKL_INT *mkrows = (MKL_INT*) calloc(nnz, sizeof(MKL_INT));
	MKL_INT *mkcols = (MKL_INT*) calloc(nnz, sizeof(MKL_INT));
	//MKL_INT mkrows[nnz];
    //MKL_INT mkcols[nnz];
    for (int i = 0; i < nnz; ++i) {
    	dvals[i] = (double) vals[i];
    	mkrows[i] = (MKL_INT) rows[i];
    	mkcols[i] = (MKL_INT) cols[i];
    }
    MKL_INT mkd = (MKL_INT) d;
    MKL_INT mkm = (MKL_INT) m;
    MKL_INT mknnz = (MKL_INT) nnz;
    sparse_matrix_t s;
    mkl_sparse_d_create_coo(&s, SPARSE_INDEX_BASE_ZERO, mkd, mkm, mknnz, mkrows, mkcols, dvals);

	// Convert the matrix to MKL's csr format.
	sparse_matrix_t s_csr;
    mkl_sparse_convert_csr(s, SPARSE_OPERATION_NON_TRANSPOSE, &s_csr);

    // apply s_csr to a to get a_hat.
    matrix_descr desc = {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT};
    if (c_major) {
		mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, s_csr, desc, SPARSE_LAYOUT_COLUMN_MAJOR,
						a, n, m, // n = # cols in output, m = leading dimension of data matrix
						0.0,
						a_hat, d);
	} else {
		mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, s_csr, desc, SPARSE_LAYOUT_ROW_MAJOR,
				a, n, n, // n = # cols in output, n = leading dimension of data matrix
				0.0,
				a_hat, n); // n = leading dimension of output matrix
	}
}

void apply_sjlt_naive_row_major(int d, int m, int n, int nnz, int *rows, int *cols, int *vals, double *a, double *a_hat) {
	// assume that a_hat has been initialized to all zeros
	//
	// implicitly have "s", a d-by-m sketching operator w/ coo data defined by (rows, cols, vals).
	// each column of s has k = nnz / m entries in (rows, cols, vals). Assuming no collisions in the row
	// indices, this should mean each column has k nonzeros.
	// assume that cols = (0,...,0, 1,...,1, 2,...,2, ...)
	int k = nnz / m;
	int idx = 0;
	int i, j, ell, _i = 0;
	double scale;
	double *a_row;
	int me, me2;

	double *dvals = (double*) calloc(nnz, sizeof(double));
    for (int i = 0; i < nnz; ++i) {
    	dvals[i] = (double) vals[i];
    }

	omp_set_num_threads(4);

	#pragma omp parallel private(ell, _i, me, me2)
	{
		//me = omp_get_thread_num();
		//me2 = omp_get_num_threads();
		//printf("Hello from %d/%d\n", me, me2);
		#pragma omp barrier

		for (ell = 0; ell < m; ++ell) {
			// NOTE: have &a[ell*m] instead of &a[ell*n] when a is m-by-m symmetric
			// when a is rectangular (row-major), should use a[ell*n]
			a_row = &a[ell*n];
			#pragma omp for
			for (_i = 0; _i < k; ++_i) {
				idx = _i + ell * k;
				scale = dvals[idx];
				i = rows[idx];
				// a_hat[i,:] += scale*a[ell,:]
				cblas_daxpy(n, scale, a_row, 1, &a_hat[i*n], 1);
				//printf("I am %d processing %d,%d\n", me, ell, _i);
				//usleep(50000);
			}
			#pragma omp barrier
		}
	}
}


void test_gen_psd() {
	int n = 3;
    double *a = (double*) calloc(n * n, sizeof(double));
    random_pos_def_matrix(n, a, 0);
    print_matrix("Test generator", n, n, a);
}


void test_gen_and_apply_sjlt_mkl(int m, int d, int n, int k) {
	// m = number of rows in a
	// d = embedding dimension; number of rows in a_hat
	// n = number of columns in a (and a_hat)
	// k = number of nonzeros per column of the sketching operator

	// OLD
    //double *a = (double*) calloc(m * m, sizeof(double)); // would like it to be m-by-n, but using random_psd.
    //random_pos_def_matrix(m, a, 0);

    double *a = (double*) calloc(n * m, sizeof(double));
    random_vector(n * m, a, 0);

	int nnz = m * k;
    int *rows = (int*) calloc(nnz, sizeof(int));
    int *cols = (int*) calloc(nnz, sizeof(int));
    int *vals = (int*) calloc(nnz, sizeof(int));
    gen_sjlt(d, m, k, rows, cols, vals, 0);

	double *a_hat = (double*) calloc(d * n, sizeof(double));
    struct timespec start, end;
    double delta_t;
	/*
    clock_gettime(CLOCK_MONOTONIC, &start);
    apply_sjlt_mkl(d, m, n, nnz, rows, cols, vals, a, a_hat, true);
	clock_gettime(CLOCK_MONOTONIC, &end);
	delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("\nMKL - column major\n");
	printf("\t%f\n", delta_t);
	*/

    //for (int i = 0; i < d*n; ++i)
    //	a_hat[i] = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    apply_sjlt_mkl(d, m, n, nnz, rows, cols, vals, a, a_hat, false);
    clock_gettime(CLOCK_MONOTONIC, &end);
	delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("MKL - row major\n");
	printf("\t%f\n", delta_t);
    //print_matrix("a_hat transpose - row major computation", n, d, a_hat);

    free(a);
    free(a_hat);
}

void test_gen_and_apply_sjlt_naive(int m, int d, int n, int k) {
	// m = number of rows in a
	// d = embedding dimension; number of rows in a_hat
	// n = number of columns in a (and a_hat)
	// k = number of nonzeros per column of the sketching operator

    //double *a = (double*) calloc(m * m, sizeof(double)); // would like it to be m-by-n, but using random_psd.
    //random_pos_def_matrix(m, a, 0);
    double *a = (double*) calloc(n * m, sizeof(double));
    random_vector(n * m, a, 0);

	int nnz = m * k;
    int *rows = (int*) calloc(nnz, sizeof(int));
    int *cols = (int*) calloc(nnz, sizeof(int));
    int *vals = (int*) calloc(nnz, sizeof(int));
    gen_sjlt(d, m, k, rows, cols, vals, 0);

    double *a_hat = (double*) calloc(d * n, sizeof(double));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    apply_sjlt_naive_row_major(d, m, n, nnz, rows, cols, vals, a, a_hat);
    clock_gettime(CLOCK_MONOTONIC, &end);
	double delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("Naive\n");
	printf("\t%f\n", delta_t);
    // print_matrix("a_hat transpose - row major computation", n, d, a_hat);

    free(a);
    free(a_hat);
}

int main() {
	int m = 100000;
	int d = 6000;
	int k = 8;
	int n = 2000;
	test_gen_and_apply_sjlt_mkl(m, d, n, k);
	test_gen_and_apply_sjlt_naive(m, d, n, k);
}



