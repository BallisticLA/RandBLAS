#ifndef randblas_sparse_hh
#define randblas_sparse_hh

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"

#include <blas.hh>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#define MAX(a, b) ((a) < (b)) ? (b) : (a)

namespace RandBLAS::sparse {

using namespace RandBLAS::base;

enum class SparseDistName : char {
    SASO = 'S',      // short-axis-sparse operator
    LASO = 'L'       // long-axis-sparse operator
};

struct SparseDist {
    const SparseDistName family = SparseDistName::SASO;
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
};

template <typename T>
struct SparseSkOp {
    const SparseDist dist;
    const base::RNGState seed_state; // maybe "self_seed"
    base::RNGState next_state; // maybe "next_seed"
    const bool own_memory = true;
    
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    int64_t *rows = nullptr;
    int64_t *cols = nullptr;
    T *vals = nullptr;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    SparseSkOp(
        SparseDist dist_,
        const base::RNGState &state_,
        int64_t *rows_ = nullptr,
        int64_t *cols_ = nullptr,
        T *vals_ = nullptr 
    );

    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows,
        int64_t *cols,
        T *vals 
    ) : SparseSkOp(dist, base::RNGState{ctr_offset, key}, rows, cols, vals) {};
    
    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        int64_t vec_nnz,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows = nullptr,
        int64_t *cols = nullptr,
        T *vals = nullptr 
    ) : SparseSkOp(SparseDist{family, n_rows, n_cols, vec_nnz},
        key, ctr_offset, rows, cols, vals) {};

    //  Destructor
    ~SparseSkOp();
};

// Implementation of elementary constructor
template <typename T>
SparseSkOp<T>::SparseSkOp(
    SparseDist dist_,
    const base::RNGState &state_,
    int64_t *rows_,
    int64_t *cols_,
    T *vals_
) :  // variable definitions
    dist(dist_),
    seed_state(state_),
    own_memory(!rows_ && !cols_ && !vals_)
{   // Initialization logic
    //
    //      own_memory is a bool that's true iff the
    //      rows_, cols_, and vals_ pointers were all nullptr.
    //
    if (this->own_memory) {
        int64_t nnz = this->dist.vec_nnz * this->dist.n_cols;
        this->rows = new int64_t[nnz];
        this->cols = new int64_t[nnz];
        this->vals = new T[nnz];
    } else {
        randblas_require(rows_ && cols_ && vals_);
        //  If any of rows_, cols_, and vals_ are not nullptr,
        //  then none of them are nullptr.
        this->rows = rows_;
        this->cols = cols_;
        this->vals = vals_;
    }
    // Implementation limitations
    randblas_require(this->dist.n_rows <= this->dist.n_cols);
};

template <typename T>
SparseSkOp<T>::~SparseSkOp() {
    if (this->own_memory) {
        delete [] this->rows;
        delete [] this->cols;
        delete [] this->vals;
    }
};

template <typename T, typename T_gen>
static RNGState template_fill_saso(
    SparseSkOp<T>& S0
) {
    randblas_require(S0.dist.family == SparseDistName::SASO);
    randblas_require(S0.dist.n_rows <= S0.dist.n_cols);
    RNGState init_state = S0.seed_state;

    // Load shorter names into the workspace
    int64_t k = S0.dist.vec_nnz;
    int64_t sa_len = S0.dist.n_rows; // short-axis length
    int64_t la_len = S0.dist.n_cols; // long-axis length
    T *vals = S0.vals; // point to array of length nnz
    int64_t *la_idxs = S0.cols; // indices of nonzeros for the long-axis
    int64_t *sa_idxs = S0.rows; // indices of nonzeros for the short-axis

    // Define variables needed in the main loop
    int64_t i, j, ell, swap, offset;
    std::vector<int64_t> pivots(k);
    std::vector<int64_t> sa_vec_work(sa_len); // short-axis vector workspace
    for (j = 0; j < sa_len; ++j) {
        sa_vec_work[j] = j;
    }
    T_gen g;
    typedef typename T_gen::ctr_type ctr_type;
    ctr_type rout;

    RNGState out_state(init_state);

    // Use Fisher-Yates
    for (i = 0; i < la_len; ++i) {
        offset = i * k;

        Random123_RNGState<T_gen> impl_state(init_state);
        impl_state.ctr.incr(offset);
        for (j = 0; j < k; ++j) {
            // one step of Fisher-Yates shuffling
            rout = g(impl_state.ctr, impl_state.key);
            ell = j + rout.v[0] % (sa_len - j);
            pivots[j] = ell;
            swap = sa_vec_work[ell];
            sa_vec_work[ell] = sa_vec_work[j];
            sa_vec_work[j] = swap;

            // update (rows, cols, vals)
            sa_idxs[j + offset] = swap;
            vals[j + offset] = (rout.v[1] % 2 == 0) ? 1.0 : -1.0;
            la_idxs[j + offset] = i;

            // increment counter
            impl_state.ctr.incr(1);
        }
        // Restore sa_vec_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes it easier to generate submatrices of
        //      a given SparseSkOp.
        for (j = 1; j <= k; ++j) {
            int jj = k - j;
            swap = sa_idxs[jj + offset];
            ell = pivots[jj];
            sa_vec_work[jj] = sa_vec_work[ell];
            sa_vec_work[ell] = swap;
        }

        out_state = impl_state;
    }
    S0.next_state = out_state;
    return out_state;
}


template <typename T>
RNGState fill_saso(
    SparseSkOp<T> &S0
) {
    switch (S0.seed_state.rng_name) {
        case RNGName::Philox:
            return template_fill_saso<T, Philox>(S0);
        case RNGName::Threefry:
            return template_fill_saso<T, Threefry>(S0);
        default:
            throw std::runtime_error(std::string("Unrecognized generator."));
    }
}

template <typename T>
void print_saso(SparseSkOp<T>& S0) {
    std::cout << "SASO information" << std::endl;
    std::cout << "\tn_rows = " << S0.dist.n_rows << std::endl;
    std::cout << "\tn_cols = " << S0.dist.n_cols << std::endl;
    int64_t nnz = S0.dist.vec_nnz * std::min(S0.dist.n_rows, S0.dist.n_cols);
    std::cout << "\tvector of row indices\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << S0.rows[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of column indices\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << S0.cols[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of values\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << S0.vals[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
static void sketch_cscrow(
    int64_t d,
    int64_t n,
    int64_t m,
    SparseSkOp<T>& S0,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T *B,
    int64_t ldb
) {
#if defined(RandBLAS_HAS_OpenMP)
    int threads = omp_get_num_threads();
#else
    int threads = 1;
#endif

    RandBLAS::sparse::SparseDist D = S0.dist;
    int64_t S_row_start = i_os;
    int64_t S_col_start = j_os;
    int64_t S_col_end = S_col_start + m;

    // Identify the range of rows to be processed by each thread.
    // TODO: replace threads = MIN(threads, d) ?
    int64_t rows_per_thread = std::max(d / threads, 1ll);
    int64_t *S_row_blocks = new int64_t[threads + 1];
    S_row_blocks[0] = S_row_start;
    for(int i = 1; i < threads + 1; ++i)
        S_row_blocks[i] = S_row_blocks[i - 1] + rows_per_thread;
    S_row_blocks[threads] = d + S_row_start;

#if defined(RandBLAS_HAS_OpenMP)
    omp_set_num_threads(threads);
    #pragma omp parallel default(shared)
    {
        // Setup variables for the current thread
        int my_id = omp_get_thread_num();
#else
        int my_id = 0;
#endif
        int64_t outer, c, offset, r, inner, S_row;
        const T *A_row = nullptr;
        T *B_row = nullptr;
        T scale;
        // Do the work for the current thread
#if defined(RandBLAS_HAS_OpenMP)
        #pragma omp for schedule(static)
#endif
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
#if defined(RandBLAS_HAS_OpenMP)
    }
#endif
}

template <typename T>
static void allrows_saso_csc_matvec(
    const T *v,
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
    const T *v,
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
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T *B,
    int64_t ldb
){
    RandBLAS::sparse::SparseDist D = S0.dist;
    int64_t r0 = i_os;
    int64_t c0 = j_os;
    int64_t rf = r0 + d;
    int64_t cf = c0 + m;
    bool all_rows_S0 = (r0 == 0 && rf == D.n_rows);

#if defined(RandBLAS_HAS_OpenMP)
    #pragma omp parallel default(shared)
    {
#endif
        // Setup variables for the current thread
        const T *A_col = nullptr;
        T *B_col = nullptr;

        // Do the work for the current thread.
#if defined(RandBLAS_HAS_OpenMP)
        #pragma omp for schedule(static)
#endif
        for (int64_t k = 0; k < n; k++) {
            A_col = &A[lda * k];
            B_col = &B[ldb * k];
            if (all_rows_S0) {
                allrows_saso_csc_matvec<T>(A_col, B_col, S0, c0, cf);
            } else {
                somerows_saso_csc_matvec<T>(A_col, B_col, S0, c0, cf, r0, rf);
            }
        }
#if defined(RandBLAS_HAS_OpenMP)
    }
#endif
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
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    randblas_require(S0.dist.family == SparseDistName::SASO);
    randblas_require(S0.rows != nullptr); // must be filled.
    randblas_require(d <= m);
    randblas_require(alpha == 1.0); // implementation limitation
    randblas_require(beta == 0.0); // implementation limitation

    // Dimensions of A, rather than op(A)
    int64_t rows_A, cols_A, rows_S, cols_S;
    SET_BUT_UNUSED(rows_S); // TODO -- implement check on rows_s and cols_s
    SET_BUT_UNUSED(cols_S);
    if (transA == blas::Op::NoTrans) {
        rows_A = m;
        cols_A = n;
    } else {
        randblas_require(false); // Not implemented.
        //rows_A = n;
        //cols_A = m;
    }
    // Dimensions of S, rather than op(S)
    if (transS == blas::Op::NoTrans) {
        rows_S = d;
        cols_S = m;
    } else {
        randblas_require(false);  // Not implemented.
        // rows_S = m;
        // cols_S = d;
    }

    // Dimensionality sanity checks, and perform the sketch.
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
        sketch_csccol<T>(d, n, m, S0, i_os, j_os, A, lda, B, ldb);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
        sketch_cscrow<T>(d, n, m, S0, i_os, j_os, A, lda, B, ldb);
    }
    return;
}

} // end namespace RandBLAS::sparse_ops

#endif
