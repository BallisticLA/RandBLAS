#include "sasos.hh"

#include <iostream>
#include <stdio.h>
#include <omp.h>

#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

#define MIN(a, b) (((a) > (b)) ? (b) : (a))


namespace RandBLAS::sasos {

template <typename T>
void fill_saso(SASO<T>& sas) {
    assert(sas.dist.n_rows <= sas.dist.n_cols);
    uint64_t seed_ctr = sas.ctr_offset;
    uint64_t seed_key = sas.key;

    // Load shorter names into the workspace
    int64_t k = sas.dist.vec_nnz;
    int64_t sa_len = sas.dist.n_rows; // short-axis length
    int64_t la_len = sas.dist.n_cols; // long-axis length
    T *vals = sas.vals; // point to array of length nnz 
    int64_t *la_idxs = sas.cols; // indices of nonzeros for the long-axis
    int64_t *sa_idxs = sas.rows; // indices of nonzeros for the short-axis

    // Define variables needed in the main loop
    int64_t i, j, ell, swap, offset;
    std::vector<int64_t> pivots(k);
    std::vector<int64_t> sa_vec_work(sa_len); // short-axis vector workspace
    for (j = 0; j < sa_len; ++j) {
        sa_vec_work[j] = j;
    }
    typedef r123::Threefry2x64 CBRNG;
	CBRNG::key_type key = {{seed_key}};
	CBRNG::ctr_type ctr = {{seed_ctr, 0}};
    CBRNG::ctr_type randpair;
	CBRNG g;

    // Use Fisher-Yates
    for (i = 0; i < la_len; ++i)
    {
        offset = i * k;
        for (j = 0; j < k; ++j)
        {
            // one step of Fisher-Yates shuffling
            ctr[0] = seed_ctr + offset + j;
            randpair = g(ctr, key);
            ell = j + randpair.v[0] % (sa_len - j);            
            pivots[j] = ell;
            swap = sa_vec_work[ell];
            sa_vec_work[ell] = sa_vec_work[j];
            sa_vec_work[j] = swap;
                   
            // update (rows, cols, vals)
            sa_idxs[j + offset] = swap;
            vals[j + offset] = (randpair.v[1] % 2 == 0) ? 1.0 : -1.0;      
            la_idxs[j + offset] = i;
        }
        // Restore sa_vec_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes it easier to generate submatrices of 
        //      a given SASO.
        for (j = 1; j <= k; ++j)
        {
            int jj = k - j;
            swap = sa_idxs[jj + offset];
            ell = pivots[jj];
            sa_vec_work[jj] = sa_vec_work[ell];
            sa_vec_work[ell] = swap;
        }
    }
    return;
}

template <typename T>
void print_saso(SASO<T>& sas)
{
    std::cout << "SASO information" << std::endl;
    std::cout << "\tn_rows = " << sas.dist.n_rows << std::endl;
    std::cout << "\tn_cols = " << sas.dist.n_cols << std::endl;
    int64_t nnz = sas.dist.vec_nnz * MIN(sas.dist.n_rows, sas.dist.n_cols);
    std::cout << "\tvector of row indices\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << sas.rows[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of column indices\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << sas.cols[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of values\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << sas.vals[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
void sketch_cscrow(
    SASO<T>& sas,
    int64_t n,
    T *a, // todo: make this const.
    int64_t lda,
    T *a_hat,
    int64_t lda_hat,
    int threads
){
    assert(lda >= n);
    assert(lda_hat >= n);
    RandBLAS::sasos::Dist D = sas.dist;
	// Identify the range of rows to be processed by each thread.
    int64_t avg = sas.dist.n_rows / threads;
    if (avg == 0) avg = 1; // this is unusual, but can happen in small experiments.
	int64_t blocks[threads + 1];
	blocks[0] = 0;
    for(int i = 1; i < threads + 1; ++i)
		blocks[i] = blocks[i - 1] + avg;
	blocks[threads] += (D.n_rows % threads); // add the remainder to the last element

    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
    {
        // Setup variables for the current thread
        int my_id = omp_get_thread_num();
        int64_t outer, c, offset, r, inner, row;
        T *a_row;
        T scale;
        // Do the work for the current thread
        #pragma omp for schedule(static)
		for (outer = 0; outer < threads; ++outer) {
			for(c = 0; c < D.n_cols; ++c) {
                // process column c of the sketching operator (row c of a)
				a_row = &a[c * lda];
				offset = c * D.vec_nnz;
                for (r = 0; r < D.vec_nnz; ++r) {
					inner = offset + r;
					row = sas.rows[inner];
					if(row >= blocks[my_id] && row < blocks[my_id + 1]) {
                        // only perform a write operation if the current row
                        // index falls in the block assigned to the current thread.
						scale = sas.vals[inner];
                        blas::axpy<T>(n, scale, a_row, 1, &a_hat[row * lda_hat], 1);
					}	
				} // end processing of column c
			}
		} 
	}
}

template <typename T>
void sketch_csccol(
    SASO<T>& sas,
    int64_t n,
    T *a, // todo: make this const
    int64_t lda,
    T *a_hat,
    int64_t lda_hat,
    int threads
){
    int64_t m = sas.dist.n_cols;
    int64_t d = sas.dist.n_rows;
    assert(lda >= m);
    assert(lda_hat >= d);
    int64_t vec_nnz = sas.dist.vec_nnz;

    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
	{
        // Setup variables for the current thread
        int64_t k, c, r, row;
        T *a_col;
        T scale;
        // Do the work for the current thread
		#pragma omp for schedule(static)
		for (k = 0; k < n; k++) {
            // process the k-th columns of a and a_hat.
			a_col = &a[lda * k];
			for (c = 0; c < m; c++) {
                // process column c of the sketching operator
				scale = a_col[c];
				for (r = c * vec_nnz; r < (c + 1) * vec_nnz; r++) {
                    row = sas.rows[r];
					a_hat[k * lda_hat + row] += (sas.vals[r] * scale);
				}		
			}
		}
	}
}

template void fill_saso<float>(SASO<float> &sas);
template void print_saso<float>(SASO<float> &sas);
template void sketch_cscrow<float>(SASO<float> &sas, int64_t n, float *a, int64_t lda, float *a_hat, int64_t lda_hat, int threads);
template void sketch_csccol<float>(SASO<float> &sas, int64_t n, float *a, int64_t lda, float *a_hat, int64_t lda_hat, int threads);


template void fill_saso<double>(SASO<double> &sas);
template void print_saso<double>(SASO<double> &sas);
template void sketch_cscrow<double>(SASO<double> &sas, int64_t n, double *a, int64_t lda, double *a_hat, int64_t lda_hat, int threads);
template void sketch_csccol<double>(SASO<double> &sas, int64_t n, double *a, int64_t lda, double *a_hat, int64_t lda_hat, int threads);
} // end namespace RandBLAS::sasos
