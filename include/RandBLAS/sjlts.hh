#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SJLTS_HH
#define RandBLAS_SJLTS_HH

#include <iostream>
#include <stdio.h>
#include <omp.h>

#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>


namespace RandBLAS::sjlts {

enum sjlt_orientation {ColumnWise, RowWise};

template <typename T>
struct SJLT {

    SJLT() : ori(sjlt_orientation::ColumnWise), n_rows(0), n_cols(0), vec_nnz(0), rows(nullptr), cols(nullptr), vals(nullptr) {}

    SJLT(sjlt_orientation _ori, uint64_t _n_rows, uint64_t _n_cols, uint64_t _vec_nnz);

    ~SJLT();

    sjlt_orientation ori;
    uint64_t n_rows;
    uint64_t n_cols;
    uint64_t vec_nnz;
    uint64_t *rows;
    uint64_t *cols;
    T *vals;
};

template <typename T>
SJLT<T>::SJLT(sjlt_orientation _ori, uint64_t _n_rows, uint64_t _n_cols, uint64_t _vec_nnz)
    : ori(_ori), n_rows(_n_rows), n_cols(_n_cols), vec_nnz(_vec_nnz) {

    uint64_t len = _vec_nnz * _n_cols;
    this->rows = new uint64_t [2 * len];
    this->cols = &(this->rows[len]);
    this->vals = new T [len];
}

template <typename T>
SJLT<T>::~SJLT() {
    delete [] this->rows;
    delete [] this->vals;
}

template <typename T>
void fill_colwise(const SJLT<T> &sjl, uint64_t seed_key, uint64_t seed_ctr) {
    // Use Fisher-Yates

    // Load shorter names into the workspace
    uint64_t k = sjl.vec_nnz;
    uint64_t n_rows = sjl.n_rows;
    uint64_t n_cols = sjl.n_cols;
    double *vals = sjl.vals; // point to array of length nnz
    uint64_t *cols = sjl.cols; // point to array of length nnz.
    uint64_t *rows = sjl.rows; // point to array of length nnz.

    // Define variables needed in the main loop
    uint64_t i, j, ell, swap, offset;
    uint64_t row_work[n_rows];
    for (j = 0; j < n_rows; ++j) {
        row_work[j] = j;
    }
    uint64_t pivots[k];
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
void print_sjlt(const SJLT<T> &sjl)
{
    std::cout << "SJLT information" << std::endl;
    std::cout << "\tn_rows = " << sjl.n_rows << std::endl;
    std::cout << "\tn_cols = " << sjl.n_cols << std::endl;
    uint64_t nnz;
    if (sjl.ori == ColumnWise)
    {
        std::cout << "\torientation: ColumnWise" << std::endl;
        nnz = sjl.vec_nnz * sjl.n_cols;
    }
    else
    {
        std::cout << "\tOrientation: RowWise" << std::endl;
        nnz = sjl.vec_nnz *  sjl.n_rows;
    }
    std::cout << "\tvector of row indices\n\t\t";
    for (uint64_t i = 0; i < nnz; ++i) {
        std::cout << sjl.rows[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of column indices\n\t\t";
    for (uint64_t i = 0; i < nnz; ++i) {
        std::cout << sjl.cols[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of values\n\t\t";
    for (uint64_t i = 0; i < nnz; ++i) {
        std::cout << sjl.vals[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
void sketch_cscrow(const SJLT<T> &sjl, uint64_t n, const T *a, T *a_hat, int threads){

	// Identify the range of rows to be processed by each thread.
    int avg = sjl.n_rows / threads;
    if (avg == 0) avg = 1; // this is unusual, but can happen in small experiments.
	int blocks[threads + 1];
	blocks[0] = 0;
    for(int i = 1; i < threads + 1; ++i)
		blocks[i] = blocks[i - 1] + avg;
	blocks[threads] += (sjl.n_rows % threads); // add the remainder to the last element

    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
	{
		int my_id = omp_get_thread_num();
		#pragma omp for schedule(static)
		for(int outer = 0; outer < threads; ++outer)
        {
			for(int c = 0; c < sjl.n_cols; ++c)
            {
				const T *a_row = &a[c * n];
				int offset = c * sjl.vec_nnz;
                for (int r = 0; r < sjl.vec_nnz; ++r)
                {
					int inner = offset + r;
					int row = sjl.rows[inner];
					if(row >= blocks[my_id] && row < blocks[my_id + 1])
                    {
						T scale = sjl.vals[inner];
                        blas::axpy(n, scale, a_row, 1, &a_hat[row * n], 1);
					}
				} // end processing of column c
			}
		}
	}
}

} // end namespace RandBLAS::sjlts

#endif // define RandBLAS_SJLTS_HH
