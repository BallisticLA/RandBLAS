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
void fill_colwise(SASO<T>& sas) {
    uint64_t seed_ctr = sas.ctr;
    uint64_t seed_key = sas.key;
    // Use Fisher-Yates

    // Load shorter names into the workspace
    int64_t k = sas.vec_nnz;
    int64_t n_rows = sas.n_rows;
    int64_t n_cols = sas.n_cols; 
    T *vals = sas.vals; // point to array of length nnz 
    int64_t *cols = sas.cols; // point to array of length nnz.
    int64_t *rows = sas.rows; // point to array of length nnz.

    // Define variables needed in the main loop
    int64_t i, j, ell, swap, offset;
    int64_t row_work[n_rows];
    for (j = 0; j < n_rows; ++j) {
        row_work[j] = j;
    }
    int64_t pivots[k];
    typedef r123::Threefry2x64 CBRNG;
	CBRNG::key_type key = {{seed_key}};
	CBRNG::ctr_type ctr = {{seed_ctr, 0}};
    CBRNG::ctr_type randpair;
	CBRNG g;

    for (i = 0; i < n_cols; ++i)
    {
        offset = i * k;
        for (j = 0; j < k; ++j)
        {
            // one step of Fisher-Yates shuffling
            ctr[0] = seed_ctr + offset + j;
            randpair = g(ctr, key);
            ell = j + randpair.v[0] % (n_rows - j);            
            pivots[j] = ell;
            swap = row_work[ell];
            row_work[ell] = row_work[j];
            row_work[j] = swap;
                   
            // update (rows, cols, vals)
            rows[j + offset] = swap;             
            if (randpair.v[1] % 2 == 0)
            {
                vals[j + offset] = 1.0;
            }
            else 
            {
                vals[j + offset] = -1.0;
            }
            cols[j + offset] = i;
        }
        // Restore row_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes debugging much easier (particularly in
        //      future parallel implementations).
        for (j = 1; j <= k; ++j)
        {
            int jj = k - j;
            swap = rows[jj + offset];
            ell = pivots[jj];
            row_work[jj] = row_work[ell];
            row_work[ell] = swap;
        }
    }
    return;
}

template <typename T>
void print_saso(SASO<T>& sas)
{
    std::cout << "SASO information" << std::endl;
    std::cout << "\tn_rows = " << sas.n_rows << std::endl;
    std::cout << "\tn_cols = " << sas.n_cols << std::endl;
    int64_t nnz = sas.vec_nnz * MIN(sas.n_rows, sas.n_cols);
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
void sketch_cscrow(SASO<T>& sas, int64_t n, T *a, T *a_hat, int threads){

	// Identify the range of rows to be processed by each thread.
    int64_t avg = sas.n_rows / threads;
    if (avg == 0) avg = 1; // this is unusual, but can happen in small experiments.
	int64_t blocks[threads + 1];
	blocks[0] = 0;
    for(int i = 1; i < threads + 1; ++i)
		blocks[i] = blocks[i - 1] + avg;
	blocks[threads] += (sas.n_rows % threads); // add the remainder to the last element

    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
	{
		int my_id = omp_get_thread_num();
		#pragma omp for schedule(static)
		for(int64_t outer = 0; outer < threads; ++outer)
        {
			for(int64_t c = 0; c < sas.n_cols; ++c)
            {
				T *a_row = &a[c * n];
				int64_t offset = c * sas.vec_nnz;
                for (int64_t r = 0; r < sas.vec_nnz; ++r)
                {
					int64_t inner = offset + r;
					int64_t row = sas.rows[inner];
					if(row >= blocks[my_id] && row < blocks[my_id + 1])
                    {
						T scale = sas.vals[inner];
                        blas::axpy<T>(n, scale, a_row, 1, &a_hat[row * n], 1);
					}	
				} // end processing of column c
			}
		} 
	}
}

template <typename T>
void sketch_csccol(SASO<T>& sas, int64_t n, T *a, T *a_hat, int threads){
    int64_t m = sas.n_cols;
    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
	{
		#pragma omp for schedule(static)
		for(int64_t k = 0; k < n; k++){
			T *a_col = &a[m * k];
			for(int64_t c = 0; c < m; c++){
				T scale = a_col[c];
				for (int64_t r = c * sas.vec_nnz; r < (c + 1) * sas.vec_nnz; r++){
                    int64_t row = sas.rows[r];
					a_hat[k * sas.n_rows + row] += (sas.vals[r] * scale);
				}		
			}
		}
	}
}

template void fill_colwise<float>(SASO<float> &sas);
template void print_saso<float>(SASO<float> &sas);
template void sketch_cscrow<float>(SASO<float> &sas, int64_t n, float *a, float *a_hat, int threads);
template void sketch_csccol<float>(SASO<float> &sas, int64_t n, float *a, float *a_hat, int threads);


template void fill_colwise<double>(SASO<double> &sas);
template void print_saso<double>(SASO<double> &sas);
template void sketch_cscrow<double>(SASO<double> &sas, int64_t n, double *a, double *a_hat, int threads);
template void sketch_csccol<double>(SASO<double> &sas, int64_t n, double *a, double *a_hat, int threads);

} // end namespace RandBLAS::sasos
