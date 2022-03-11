#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <mkl_spblas.h>
#include <omp.h>
#include <assert.h>
#include <vector>
#include <tuple>
#include <bits/stdc++.h>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
auto t1 = high_resolution_clock::now();
auto t2 = high_resolution_clock::now();
auto t3 = duration_cast<milliseconds>(t2 - t1);

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
/*
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
*/
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

void sortArr_csr(int arr[], int n, MKL_INT *rows, MKL_INT *cols, int *dvals)
{
  
    // Vector to store element
    // with respective present index
    std::vector<std::tuple<int, int, int> > vp;
  
    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < n; ++i) {
        vp.push_back(std::tuple<int, int, int>(arr[i], cols[i], dvals[i]));
    }
  
    // Sorting pair vector
    std::sort(vp.begin(), vp.end());
  
    for (int i = 0; i < vp.size(); i++) {
        rows[i] = std::get<0>(vp[i]); 
		cols[i] = std::get<1>(vp[i]); 
		dvals[i] = std::get<2>(vp[i]); 
		
    }
}


void sortArr_csr(int arr[], int n, MKL_INT *rows, MKL_INT *cols, double *dvals)
{
  
    // Vector to store element
    // with respective present index
    std::vector<std::tuple<int, int, double> > vp;
  
    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < n; ++i) {
        vp.push_back(std::tuple<int, int, double>(arr[i], cols[i], dvals[i]));
    }
  
    // Sorting pair vector
    std::sort(vp.begin(), vp.end());
  
    for (int i = 0; i < vp.size(); i++) {
        rows[i] = std::get<0>(vp[i]); 
		cols[i] = std::get<1>(vp[i]); 
		dvals[i] = std::get<2>(vp[i]); 
		
    }
}

void create_csr_from_coo(MKL_INT d, MKL_INT m, MKL_INT nnz, MKL_INT *mkrows, 
	MKL_INT *mkcols, double *dvals, sparse_matrix_t *output){
	assert(nnz != 0);
	sortArr_csr(mkrows, nnz, mkrows, mkcols, dvals); // sort based on row index
	// allocate space for new sparse matrix
	double *csr_vals = (double*) calloc(nnz, sizeof(double));
	MKL_INT *csr_rows_start = (MKL_INT*) calloc(d, sizeof(MKL_INT));
	MKL_INT *csr_rows_end = (MKL_INT*) calloc(d, sizeof(MKL_INT));
	MKL_INT *csr_cols = (MKL_INT*) calloc(nnz, sizeof(MKL_INT));
	
	MKL_INT prev_row = mkrows[0];
	MKL_INT count = 0;
	MKL_INT prev_val = 0;
	for (int i = 0; i < nnz; i++){
		csr_cols[i] = mkcols[i];
		csr_vals[i] = dvals[i];
		if(mkrows[i] == prev_row){
			count++;
		}else{
			csr_rows_start[prev_row] = prev_val;
			prev_val += count;
			csr_rows_end[prev_row] = prev_val;
			for(int j = prev_row + 1; j < mkrows[i]; j++){
				csr_rows_start[j] = prev_val;
				csr_rows_end[j] = prev_val;
			}
			count = 1;
			prev_row = mkrows[i];
		}
	}
	// remainder
	csr_rows_start[prev_row] = prev_val;
	prev_val += count;
	csr_rows_end[prev_row] = prev_val;
	for(int j = prev_row + 1; j < d; j++){
		csr_rows_start[j] = prev_val;
		csr_rows_end[j] = prev_val;
	}

	int status = mkl_sparse_d_create_csr(output, SPARSE_INDEX_BASE_ZERO, d, m, csr_rows_start, csr_rows_end, csr_cols, csr_vals);
	/*
	for (int i = 0; i < nnz; i++){
		std::cout << "csr_cols: " << csr_cols[i] << " ";
		std::cout << "csr_vals: " << csr_vals[i] << " ";
		std::cout << "\n";
	}
	for (int i = 0; i < d; i++){
		std::cout << "csr_rows_start: " << csr_rows_start[i] << " ";
		std::cout << "csr_rows_end: " << csr_rows_end[i] << " ";
		std::cout << "\n";
	}
	*/
	assert(status == SPARSE_STATUS_SUCCESS);

}

void sortArr_csc(int arr[], int n, MKL_INT *rows, MKL_INT *cols, double *dvals)
{
  
    // Vector to store element
    // with respective present index
    std::vector<std::tuple<int, int, double> > vp;
  
    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < n; ++i) {
        vp.push_back(std::tuple<int, int, double>(arr[i], rows[i], dvals[i]));
    }
  
    // Sorting pair vector
    std::sort(vp.begin(), vp.end());
  
    for (int i = 0; i < vp.size(); i++) {
        cols[i] = std::get<0>(vp[i]); 
		rows[i] = std::get<1>(vp[i]); 
		dvals[i] = std::get<2>(vp[i]); 
		
    }
}

void sortArr_csc(int arr[], int n, MKL_INT *rows, MKL_INT *cols, int *dvals)
{
  
    // Vector to store element
    // with respective present index
    std::vector<std::tuple<int, int, int> > vp;
  
    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < n; ++i) {
        vp.push_back(std::tuple<int, int, int>(arr[i], rows[i], dvals[i]));
    }
  
    // Sorting pair vector
    std::sort(vp.begin(), vp.end());
  
    for (int i = 0; i < vp.size(); i++) {
        cols[i] = std::get<0>(vp[i]); 
		rows[i] = std::get<1>(vp[i]); 
		dvals[i] = std::get<2>(vp[i]); 
		
    }
}

void create_csc_from_coo(MKL_INT d, MKL_INT m, MKL_INT nnz, MKL_INT *mkrows, 
	MKL_INT *mkcols, int *vals, sparse_matrix_t *output){
	assert(nnz != 0);
	sortArr_csc(mkcols, nnz, mkrows, mkcols, vals); // sort based on col index

	

	// allocate space for new sparse matrix
	double *dvals = (double*) calloc(nnz, sizeof(double));
	for (int i = 0; i < nnz; ++i) {
		dvals[i] = (double) vals[i];
    }
	MKL_INT *csr_cols_start = (MKL_INT*) calloc(m, sizeof(MKL_INT));
	MKL_INT *csr_cols_end = (MKL_INT*) calloc(m, sizeof(MKL_INT));
	MKL_INT *csr_rows = (MKL_INT*) calloc(nnz, sizeof(MKL_INT));
	
	MKL_INT prev_col = mkcols[0];
	MKL_INT count = 0;
	MKL_INT prev_val = 0;
	for (int i = 0; i < nnz; i++){
		csr_rows[i] = mkrows[i];
		if(mkcols[i] == prev_col){
			count++;
		}else{
			csr_cols_start[prev_col] = prev_val;
			prev_val += count;
			csr_cols_end[prev_col] = prev_val;
			for(int j = prev_col + 1; j < mkcols[i]; j++){
				csr_cols_start[j] = prev_val;
				csr_cols_end[j] = prev_val;
			}
			count = 1;
			prev_col = mkcols[i];
		}
	}
	// remainder
	csr_cols_start[prev_col] = prev_val;
	prev_val += count;
	csr_cols_end[prev_col] = prev_val;
	for(int j = prev_col + 1; j < m; j++){
		csr_cols_start[j] = prev_val;
		csr_cols_end[j] = prev_val;
	}

	int status = mkl_sparse_d_create_csc(output, SPARSE_INDEX_BASE_ZERO, d, m, csr_cols_start, csr_cols_end, csr_rows, dvals);
	/*
	for (int i = 0; i < nnz; i++){
		std::cout << "csr_cols: " << csr_cols[i] << " ";
		std::cout << "csr_vals: " << csr_vals[i] << " ";
		std::cout << "\n";
	}
	for (int i = 0; i < d; i++){
		std::cout << "csr_rows_start: " << csr_rows_start[i] << " ";
		std::cout << "csr_rows_end: " << csr_rows_end[i] << " ";
		std::cout << "\n";
	}
	*/
	assert(status == SPARSE_STATUS_SUCCESS);

}


// sparse matrix times dense matrix gives dense matrix
// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/inspector-executor-sparse-blas-execution-routines/mkl-sparse-mm.html

void gen_sjlt_with_repeat(int d, int m, int k, int *rows, int *cols, int *vals, int seed) {
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


void fisher_yates_sample(int *fill, int k, int* indices, int index_length, VSLStreamStatePtr &stream){
	for (int i = 0; i < k; i++){
		int num;
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream,
					 1, // number of samples
					 &num, // place to write sample
				 	 0, // lower bound (inclusive)
				  	 index_length - i // upper bound (exclusive)
		);
		//int num = rand() % (index_length - i);
		fill[i] = indices[num];
		// swap the two indices
		indices[num] = indices[index_length - i - 1];
		indices[index_length - i - 1] = fill[i];
	}
}

void gen_sjlt(int d, int m, int k, int *rows, int *cols, int* vals, int seed){
	int nnz = m * k;
	int *indices = new int[d];
	// generate 1 or -1 for val
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);
	

	for (int i = 0; i < d; i++){
		indices[i] = i;
	}
	// sample index using whatever method
	for (int c = 0; c < m; c++){
		fisher_yates_sample(&rows[c * k], k, indices, d, stream);
		for(int j = 0; j < k; j++){
			cols[c * k + j] = c;
		}
	}

	// generate 1 or -1
	viRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream, nnz, vals, 0, 2);
	vslDeleteStream(&stream);
	for (int i = 0; i < nnz; ++i) {
		vals[i] = 2*vals[i] - 1;
	}
	free(indices);
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

/*
	for (int i = 0; i < mknnz; i++){
		std::cout << "rows: " << mkrows[i] << " ";
		std::cout << "cols: " << mkcols[i] << " ";
		std::cout << "dvals: " << dvals[i] << " ";
		std::cout << "\n";
	}
	
	
	MKL_INT pointerB[5] = {0,3,5,8,11};
	MKL_INT pointerE[5] = {3,5,8,11,13};
	MKL_INT rr[13] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
	MKL_INT columns[13] = {0,1,3,0,1,2,3,4,0,2,3,1,4};
	double values[13] = {0,1,3,0,1,2,3,4,0,2,3,1,4};
	sparse_matrix_t test;
	sparse_matrix_t test1;
	int *aa = new int[2];
	int *bb = new int[2];
	double *cc = new double[2];
	aa[0] = 0;
	bb[0] = 0;
	cc[0] = 1.0;
	int ret = mkl_sparse_d_create_coo(&test, SPARSE_INDEX_BASE_ZERO, 5, 5, 1, aa, bb, cc);
	//mkl_sparse_d_create_csr(&test, 
	//	SPARSE_INDEX_BASE_ZERO, 5, 5, pointerB, pointerE, columns, values);
	std::cout << ret << "\n";
	assert(ret == SPARSE_STATUS_SUCCESS);
	mkl_sparse_convert_csr(test, SPARSE_OPERATION_NON_TRANSPOSE, &test1);
	*/
	std::cout << mkd << "\n";
	std::cout << mkm << "\n";
	std::cout << mknnz << "\n";
	
	
	// Convert the matrix to MKL's csr format.
	sparse_matrix_t s;
    mkl_sparse_d_create_coo(&s, SPARSE_INDEX_BASE_ZERO, mkd, mkm, mknnz, mkrows, mkcols, dvals);

	// Convert the matrix to MKL's csr format.
	sparse_matrix_t s_csr;
    mkl_sparse_convert_csr(s, SPARSE_OPERATION_NON_TRANSPOSE, &s_csr);

	

/*
	sparse_matrix_t check;
	mkl_sparse_convert_csr(s_csc, SPARSE_OPERATION_NON_TRANSPOSE, &check);


	MKL_INT r = 0;
	MKL_INT c = 0;
	MKL_INT *pointerB;
	MKL_INT *pointerE;
	MKL_INT *pointer_col;
	sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
	double *pointerval;
	mkl_sparse_d_export_csr(s_csr, &indexing, &r, &c, &pointerB, &pointerE, &pointer_col, &pointerval);

	MKL_INT r1 = 0;
	MKL_INT c1 = 0;
	MKL_INT *pointerB1;
	MKL_INT *pointerE1;
	MKL_INT *pointer_col1;
	sparse_index_base_t indexing1 = SPARSE_INDEX_BASE_ZERO;
	double *pointerval1;
	mkl_sparse_d_export_csr(check, &indexing1, &r1, &c1, &pointerB1, &pointerE1, &pointer_col1, &pointerval1);
	for (int i = 0; i < r; i++){
		assert(pointerB1[i] == pointerB[i]);
		assert(pointerE1[i] == pointerE[i]);
	}
	for (int i = 0; i < nnz; i++){
		assert(pointer_col[i] == pointer_col1[i]);
		assert(pointerval[i] == pointerval1[i]);
	}
*/
	//std::cout << "MKL max threads: " << mkl_get_max_threads() << "\n";
	//mkl_set_num_threads(omp_get_max_threads());
    // apply s_csr to a to get a_hat.
    matrix_descr desc = {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT};
    if (c_major) {
		t1 = high_resolution_clock::now();
		mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, s_csr, desc, SPARSE_LAYOUT_COLUMN_MAJOR,
						a, n, m, // n = # cols in output, m = leading dimension of data matrix
						0.0,
						a_hat, d);
		t2 = high_resolution_clock::now();
		std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms for row major/column major computation time\n";
	} else {
		

		
		t1 = high_resolution_clock::now();
		mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, s_csr, desc, SPARSE_LAYOUT_ROW_MAJOR,
				a, n, n, // n = # cols in output, n = leading dimension of data matrix
				0.0,
				a_hat, n); // n = leading dimension of output matrix
		t2 = high_resolution_clock::now();
		std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms for row major/row major computation time\n";

		sparse_matrix_t s_csc;
    	create_csc_from_coo(d, m, nnz, rows, cols, vals, &s_csc);
		t1 = high_resolution_clock::now();
		mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, s_csc, desc, SPARSE_LAYOUT_ROW_MAJOR,
				a, n, n, // n = # cols in output, n = leading dimension of data matrix
				0.0,
				a_hat, n); // n = leading dimension of output matrix
		t2 = high_resolution_clock::now();
		std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms for col major/row major computation time\n";

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

	
	//omp_set_num_threads(2);
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
				//std::cout << omp_get_thread_num() << "\n";
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

void apply_my_sjlt_csr(int d, int m, int n, int nnz, int *rows, int *cols, int *vals, double *a, double *a_hat){

	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for(int c = 0; c < d; c++){
			double *a_hat_row = &a_hat[c * n];
			for (int r = rows[c]; r < rows[c + 1]; r++){
				int col = cols[r];
				int scale = vals[r];
				cblas_daxpy(n, scale, &a[col * n], 1, a_hat_row, 1);
			}
		}
	
	
	}
	
}

void apply_my_sjlt_csc(int d, int m, int n, int nnz, int *rows, int *cols, int *vals, double *a, double *a_hat){
	int k = nnz / m;
	int max_threads = omp_get_max_threads();
	int index_range[max_threads + 1];
	index_range[0] = 0;
	int avg = d / max_threads;
	for(int i = 1; i < max_threads + 1; i++){
		index_range[i] = index_range[i - 1] + avg;
	}
	index_range[max_threads] += (d % max_threads); // add the remainder to the last element
	for(int i = 0; i < max_threads + 1; i++){
		std::cout << index_range[i] << "\n";
	}
	
	#pragma omp parallel default(shared)
	{
		int my_id = omp_get_thread_num();
		

		#pragma omp for schedule(static)
		for(int outer = 0; outer < max_threads; outer++){
			for(int c = 0; c < m; c++){
				//printf("c: %d, id: %d\n", c, omp_get_thread_num());
				double *a_row = &a[c * n];
				for (int r = 0; r < k; r++){
					int inner = c * k + r;
					int row = rows[inner];
					if(row >= index_range[my_id] && row < index_range[my_id + 1]){
						int scale = vals[inner];
						cblas_daxpy(n, scale, a_row, 1, &a_hat[row * n], 1);
					}	

				}
			}
		}
		
	}
	
}


void test_gen_psd() {
	int n = 3;
    double *a = (double*) calloc(n * n, sizeof(double));
    random_pos_def_matrix(n, a, 0);
    print_matrix("Test generator", n, n, a);
}


double * test_gen_and_apply_sjlt_mkl(int m, int d, int n, int k) {
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
	double *b_hat = (double*) calloc(d * n, sizeof(double));
    struct timespec start, end;
    double delta_t;
	
	//int d, int m, int n, int nnz, int *rows, int *cols, int *vals, double *a, double *a_hat, bool c_major
    clock_gettime(CLOCK_MONOTONIC, &start);
    apply_sjlt_mkl(d, m, n, nnz, rows, cols, vals, a, a_hat, true);
	clock_gettime(CLOCK_MONOTONIC, &end);
	delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("\nMKL - column major\n");
	printf("\t%f\n", delta_t);
	
	
    //for (int i = 0; i < d*n; ++i)
    //	a_hat[i] = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    apply_sjlt_mkl(d, m, n, nnz, rows, cols, vals, a, b_hat, false);
    clock_gettime(CLOCK_MONOTONIC, &end);
	delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("MKL - row major\n");
	printf("\t%f\n", delta_t);
    //print_matrix("a_hat transpose - row major computation", n, d, a_hat);

    free(a);
	free(a_hat);
	//free(b_hat);
	return b_hat;
}

void add_function(){
	int n = 100000000;
	double *a = (double*) calloc(n, sizeof(double));
	double *b = (double*) calloc(n, sizeof(double));
	double *c = (double*) calloc(n, sizeof(double));
	omp_set_num_threads(2);
	#pragma omp parallel for
	for(int i = 0; i < n; i++){
		c[i] = a[i] + b[i];
	}
	free(a);
	free(b);
	std::cout << c[0] << "\n";
	free(c);
}

double * test_gen_and_apply_sjlt_naive(int m, int d, int n, int k) {
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
	printf("csc parallel method 1\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
	apply_my_sjlt_csc(d, m, n, nnz, rows, cols, vals, a, a_hat);
    clock_gettime(CLOCK_MONOTONIC, &end);
	double delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("\t%f\n", delta_t);


	printf("csc parallel method 2\n");
	double *b_hat = (double*) calloc(d * n, sizeof(double));
	clock_gettime(CLOCK_MONOTONIC, &start);
	//add_function();
	apply_sjlt_naive_row_major(d, m, n, nnz, rows, cols, vals, a, b_hat);
	clock_gettime(CLOCK_MONOTONIC, &end);
	delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("\t%f\n", delta_t);
    // print_matrix("a_hat transpose - row major computation", n, d, a_hat);


	printf("csr parallel method 1\n");
	double *c_hat = (double*) calloc(d * n, sizeof(double));
	sortArr_csr(rows, nnz, rows, cols, vals); // sort based on row index
	int *row_count = (int*) calloc(d + 1, sizeof(int));
	int prev = rows[0];
	int count = 0;
	for(int i = 0; i < nnz; i++){
		if(rows[i] != prev){
			row_count[prev + 1] = count;
			for (int j = prev + 2; j < rows[i] + 1; j++){
				row_count[j] = row_count[j - 1];
			}
			prev = rows[i];
		}
		count++;
	}
	row_count[prev + 1] = count;
	for (int j = prev + 2; j < d + 1; j++){
		row_count[j] = row_count[j - 1];
	}


	
    clock_gettime(CLOCK_MONOTONIC, &start);
	apply_my_sjlt_csr(d, m, n, nnz, row_count, cols, vals, a, c_hat);
    clock_gettime(CLOCK_MONOTONIC, &end);
	delta_t = 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	delta_t /= 1e9;
	printf("\t%f\n", delta_t);


	for(int i = 0; i < d * n; i++){
		assert(abs(a_hat[i] - b_hat[i]) < 1e-13);
		assert(abs(b_hat[i] - c_hat[i]) < 1e-13);
	}

	free(row_count);
    free(a);
    //free(a_hat);
	free(b_hat);
	free(c_hat);
	return a_hat;
}



int main() {
	int m = 100000;
	int d = 6000;
	int k = 8;
	int n = 2000;
	double* a = test_gen_and_apply_sjlt_mkl(m, d, n, k);
	double* b = test_gen_and_apply_sjlt_naive(m, d, n, k);
	for(int i = 0; i < m * k; i++){
		if(abs(a[i] - b[i]) > 1e-13){
			std::cout << "i: " << i << "\n";
			std::cout << a[i] << "\n";
			std::cout << b[i] << "\n";
			assert(abs(a[i] - b[i]) < 1e-13);
		}
		

	}
	free(a);
	free(b);
}


