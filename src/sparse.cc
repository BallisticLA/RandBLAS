#include "sparse.hh"

#include <iostream>
#include <stdio.h>
#include <omp.h>

#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAX(a, b) (((a) <= (b)) ? (b) : (a))

namespace RandBLAS::sparse {

static std::pair<int64_t, int64_t> indexing_bounds(
    int64_t A0_rows,
    int64_t A0_cols,
    int64_t poA,
    blas::Layout layout
) {
    std::pair<int64_t, int64_t> out;
    // (out.first, out.second) = (row index, column index)
    if (layout == blas::Layout::ColMajor) {
        out.second = poA / A0_rows;
        out.first = poA % A0_rows;
    } else {
        out.first = poA / A0_cols;
        out.second = poA % A0_cols;
    }
    return out;
}


template <typename T>
void fill_saso(SparseSkOp<T>& sas) {
    assert(sas.dist.family == SparseDistName::SASO);
    //assert(sas.dist.dist4nz.family == RandBLAS::dense_op::DistName::Rademacher);
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
        //      a given SparseSkOp.
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
void print_saso(SparseSkOp<T>& sas)
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
static void sketch_cscrow(
    int64_t d,
    int64_t n,
    int64_t m,
    SparseSkOp<T>& S0,
    int64_t pos,
    T *A, // todo: make this const.
    int64_t lda,
    T *B,
    int64_t ldb,
    int threads
){
    RandBLAS::sparse::SparseDist D = S0.dist;
    auto starts = indexing_bounds(D.n_rows, D.n_cols, pos, blas::Layout::RowMajor);
	int64_t S_row_start = starts.first;
    int64_t S_col_start = starts.second;
    int64_t S_col_end = S_col_start + m;

    // Identify the range of rows to be processed by each thread.
    // TODO: replace threads = MIN(threads, d) ?
    int64_t rows_per_thread = MAX(d / threads, 1);
	int64_t *S_row_blocks = new int64_t[threads + 1];
	S_row_blocks[0] = S_row_start;
    for(int i = 1; i < threads + 1; ++i)
		S_row_blocks[i] = S_row_blocks[i - 1] + rows_per_thread;
	S_row_blocks[threads] = d + S_row_start;

    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
    {
        // Setup variables for the current thread
        int my_id = omp_get_thread_num();
        int64_t outer, c, offset, r, inner, S_row;
        T *A_row, *B_row;
        T scale;
        // Do the work for the current thread
        #pragma omp for schedule(static)
		for (outer = 0; outer < threads; ++outer) {
			for(c = S_col_start; c < S_col_end; ++c) {
                // process column c of the sketching operator (row c of a)
				A_row = &A[c * lda];
				offset = c * D.vec_nnz;
                for (r = 0; r < D.vec_nnz; ++r) {
					inner = offset + r;
					S_row = S0.rows[inner];
					if (
                        S_row_blocks[my_id] <= S_row && S_row < S_row_blocks[my_id + 1]
                    ) {
                        // only perform a write operation if the current row
                        // index falls in the block assigned to the current thread.
						scale = S0.vals[inner];
                        B_row = &B[(S_row - S_row_start) * ldb];
                        blas::axpy<T>(n, scale, A_row, 1, B_row, 1);
					}	
				} // end processing of column c
			}
		} 
	}
}

template <typename T>
static void allrows_saso_csc_matvec(
    T *v,
    T *Sv, // Sv += S0[:, col_start:col_end] * v.
    SparseSkOp<T> &S0,
    int64_t col_start,
    int64_t col_end
) {
    int64_t vec_nnz = S0.dist.vec_nnz;
    for (int64_t c = col_start; c < col_end; c++) {
        T scale = v[c - col_start];
        for (int64_t r = c * vec_nnz; r < (c + 1) * vec_nnz; r++) {
            int64_t S0_row = S0.rows[r];
            Sv[S0_row] += (S0.vals[r] * scale);
        }
    }
}

template <typename T>
static void somerows_saso_csc_matvec(
    T *v,
    T *Sv, // Sv += S0[row_start:row_end, col_start:col_end] * v.
    SparseSkOp<T> &S0,
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end
) {
    int64_t vec_nnz = S0.dist.vec_nnz;
    for (int64_t c = col_start; c < col_end; c++) {
        T scale = v[c - col_start];
        for (int64_t r = c * vec_nnz; r < (c + 1) * vec_nnz; r++) {
            int64_t S0_row = S0.rows[r];
            if (row_start <= S0_row && S0_row < row_end)
                Sv[S0_row - row_start] += (S0.vals[r] * scale);
        }
    }
}

template <typename T>
static void sketch_csccol(
    int64_t d,
    int64_t n,
    int64_t m,
    SparseSkOp<T>& S0,
    int64_t pos,
    T *A, // todo: make this const
    int64_t lda,
    T *B,
    int64_t ldb,
    int threads
){
    RandBLAS::sparse::SparseDist D = S0.dist;
    int64_t vec_nnz = D.vec_nnz;
    auto starts = indexing_bounds(D.n_rows, D.n_cols, pos, blas::Layout::ColMajor);
	int64_t r0 = starts.first;
    int64_t c0 = starts.second;
    int64_t rf = r0 + d;
    int64_t cf = c0 + m;
    bool all_rows_S0 = (r0 == 0 && rf == D.n_rows);

    omp_set_num_threads(threads);
	#pragma omp parallel default(shared)
	{
        // Setup variables for the current thread
        T *A_col, *B_col;
        // Do the work for the current thread.
		#pragma omp for schedule(static)
		for (int64_t k = 0; k < n; k++) {
			A_col = &A[lda * k];
            B_col = &B[ldb * k];
            if (all_rows_S0) {
                allrows_saso_csc_matvec<T>(A_col, B_col, S0, c0, cf);
            } else {
                somerows_saso_csc_matvec<T>(A_col, B_col, S0, c0, cf, r0, rf);
            }
		}
	}
}

template <typename T>
void lskges(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    SparseSkOp<T> &S0,
    int64_t i_os,
    int64_t j_os,
    T *A, // TODO: make const
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb,
    int threads // default is 4.
) {
    assert(S0.dist.family == SparseDistName::SASO);
    assert(S0.rows != NULL); // must be filled.
    assert(d <= m);
    assert(alpha == 1.0); // implementation limitation
    assert(beta == 0.0); // implementation limitation

    // Dimensions of A, rather than op(A)
    int64_t rows_A, cols_A, rows_S, cols_S;
    if (transA == blas::Op::NoTrans) {
        rows_A = m;
        cols_A = n;
    } else {
        assert(false); // Not implemented.
        //rows_A = n;
        //cols_A = m;
    }
    // Dimensions of S, rather than op(S)
    if (transS == blas::Op::NoTrans) {
        rows_S = d;
        cols_S = m;
    } else {
        assert(false);  // Not implemented.
        // rows_S = m;
        // cols_S = d;
    }
    
    // Dimensionality sanity checks, and perform the sketch.
    if (layout == blas::Layout::ColMajor) {
        int64_t lds = S0.dist.n_rows;
        int64_t pos = i_os + lds * j_os;
        assert(lds >= rows_S);
        assert(lda >= rows_A);
        assert(ldb >= d);
        sketch_csccol<T>(d, n, m, S0, pos, A, lda, B, ldb, threads);
    } else {
        int64_t lds = S0.dist.n_cols;
        int64_t pos = i_os * lds + j_os;
        assert(lds >= cols_S);
        assert(lda >= cols_A);
        assert(ldb >= n);
        sketch_cscrow<T>(d, n, m, S0, pos, A, lda, B, ldb, threads);
    }
    return;
}


template void fill_saso<float>(SparseSkOp<float> &sas);
template void print_saso<float>(SparseSkOp<float> &sas);
template void sketch_cscrow<float>(int64_t d, int64_t n, int64_t m, SparseSkOp<float> &S0, int64_t pos, float *A, int64_t lda, float *B, int64_t ldb, int threads);
template void sketch_csccol<float>(int64_t d, int64_t n, int64_t m, SparseSkOp<float> &S0, int64_t pos, float *A, int64_t lda, float *B, int64_t ldb, int threads);
template void lskges<float>(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, float alpha,
    SparseSkOp<float> &S0, int64_t i_os, int64_t j_os, float *A, int64_t lda, float beta, float *B, int64_t ldb, int threads);


template void fill_saso<double>(SparseSkOp<double> &sas);
template void print_saso<double>(SparseSkOp<double> &sas);
template void sketch_cscrow<double>(int64_t d, int64_t n, int64_t m, SparseSkOp<double> &S0, int64_t pos, double *A, int64_t lda, double *B, int64_t ldb, int threads);
template void sketch_csccol<double>(int64_t d, int64_t n, int64_t m, SparseSkOp<double> &S0, int64_t pos, double *A, int64_t lda, double *B, int64_t ldb, int threads);
template void lskges<double>(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, double alpha,
    SparseSkOp<double> &S0, int64_t i_os, int64_t j_os, double *A, int64_t lda, double beta, double *B, int64_t ldb, int threads);

} // end namespace RandBLAS::sparse_ops