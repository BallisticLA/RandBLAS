#include <RandBLAS/exceptions.hh>
#include <RandBLAS/sparse.hh>

#include <iostream>
#include <stdio.h>
#include <omp.h>

#include <Random123/uniform.hpp>

#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAX(a, b) (((a) <= (b)) ? (b) : (a))

namespace RandBLAS::sparse {

using namespace RandBLAS::base;

template <typename T>
static bool fixed_nnz_per_col(
    SparseSkOp<T> &S0
) {
    if (S0.dist.family == SparseDistName::SASO) {
        return S0.dist.n_rows < S0.dist.n_cols;
    } else {
        return S0.dist.n_cols < S0.dist.n_rows;
    }
}

template <typename T, typename T_gen>
static RNGState repeated_fisher_yates(
    RNGState init_state,
    int64_t num_vecs,
    int64_t vec_len,
    int64_t vec_nnz,
    int64_t *vec_ax_idxs,
    int64_t *rep_ax_idxs,
    T *vals
) {
    std::vector<int64_t> vec_work(vec_len);
    for (int64_t j = 0; j < vec_len; ++j)
        vec_work[j] = j;
    std::vector<int64_t> pivots(vec_nnz);
    typedef typename T_gen::ctr_type ctr_type;
    ctr_type rout;
    T_gen g;
    RNGState out_state(init_state);
    for (int64_t i = 0; i < num_vecs; ++i) {
        // set the state of the Random123 RNG.
        int64_t offset = i * vec_nnz;
        Random123_RNGState<T_gen> impl_state(init_state);
        impl_state.ctr.incr(offset);
        for (int64_t j = 0; j < vec_nnz; ++j) {
            // one step of Fisher-Yates shuffling
            rout = g(impl_state.ctr, impl_state.key);
            int64_t ell = j + rout.v[0] % (vec_len - j);
            pivots[j] = ell;
            int64_t swap = vec_work[ell];
            vec_work[ell] = vec_work[j];
            vec_work[j] = swap;
            // update (rows, cols, vals)
            vec_ax_idxs[j + offset] = swap;
            vals[j + offset] = (rout.v[1] % 2 == 0) ? 1.0 : -1.0;
            rep_ax_idxs[j + offset] = i;
            // increment counter
            impl_state.ctr.incr(1);
        }
        // Restore vec_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes it easier to generate submatrices of
        //      a given SparseSkOp.
        for (int64_t j = 1; j <= vec_nnz; ++j) {
            int64_t jj = vec_nnz - j;
            int64_t swap = vec_ax_idxs[jj + offset];
            int64_t ell = pivots[jj];
            vec_work[jj] = vec_work[ell];
            vec_work[ell] = swap;
        }
        out_state = impl_state;
    }
    return out_state;
}

template <typename T, typename T_gen>
static RNGState template_fill_sparse(
    SparseSkOp<T>& S0
) {
    int64_t long_ax_len = MAX(S0.dist.n_rows, S0.dist.n_cols);
    int64_t short_ax_len = MIN(S0.dist.n_rows, S0.dist.n_cols);

    bool is_wide = S0.dist.n_rows == short_ax_len;
    int64_t *short_ax_idxs = (is_wide) ? S0.rows : S0.cols;
    int64_t *long_ax_idxs = (is_wide) ? S0.cols : S0.rows;

    int64_t *vec_ax_idxs, *rep_ax_idxs;
    int64_t vec_len, num_vecs;
    if (S0.dist.family == SparseDistName::SASO) {
        vec_len = short_ax_len;
        num_vecs = long_ax_len;
        vec_ax_idxs = short_ax_idxs;
        rep_ax_idxs = long_ax_idxs;
    } else {
        vec_len = long_ax_len;
        num_vecs = short_ax_len;
        vec_ax_idxs = long_ax_idxs;
        rep_ax_idxs = short_ax_idxs;
    }
    RNGState out_state = repeated_fisher_yates<T, T_gen>(
        S0.seed_state, num_vecs, vec_len,
        S0.dist.vec_nnz, vec_ax_idxs, rep_ax_idxs, S0.vals
    );
    S0.next_state = out_state;
    return out_state;
}

template RNGState template_fill_sparse<float, Philox>(SparseSkOp<float> &S0);
template RNGState template_fill_sparse<double, Philox>(SparseSkOp<double> &S0);
template RNGState template_fill_sparse<float, Threefry>(SparseSkOp<float> &S0);
template RNGState template_fill_sparse<double, Threefry>(SparseSkOp<double> &S0);

template <typename T>
RNGState fill_sparse(
    SparseSkOp<T> &S0
) {
    switch (S0.seed_state.rng_name) {
        case RNGName::Philox:
            return template_fill_sparse<T, Philox>(S0);
        case RNGName::Threefry:
            return template_fill_sparse<T, Threefry>(S0);
        default:
            throw std::runtime_error(std::string("Unrecognized generator."));
    }
}

template <typename T>
void print_saso(SparseSkOp<T>& S0) {
    std::cout << "SASO information" << std::endl;
    std::cout << "\tn_rows = " << S0.dist.n_rows << std::endl;
    std::cout << "\tn_cols = " << S0.dist.n_cols << std::endl;
    int64_t nnz = S0.dist.vec_nnz * MIN(S0.dist.n_rows, S0.dist.n_cols);
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
static int64_t filter_regular_cscoo(
    const int64_t *nzidx2row,
    const int64_t *nzidx2col,
    const T *nzidx2val,
    int64_t vec_nnz,
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end,
    int64_t *rows_view,
    int64_t *cols_view,
    T *vals_view
) {
    // (nzidx2row, nzidx2col, nzidx2val) define a sparse matrix S0
    //  in COO format with a CSC-like property. The CSC-like property
    //  is that all nonzero entries in a given column of S0 are 
    //  contiguous in "nzidx2col". The sparse matrix S0 must be "regular"
    //  in the sense that it must have exactly vec_nnz nonzeros in each
    //  column.
    //
    //  This function writes data to (rows_view, cols_view, vals_view)
    //  for a sparse matrix S in the same "COO/CSC-like" format.
    //  Mathematically, we have S = S0[row_start:row_end, col_start:col_end].
    //  The sparse matrix S does not necessarily have a fixed number of
    //  nonzeros in each column.
    //
    //  Neither S0 nor S need to be wide.  
    //
    int64_t nnz = 0;
    for (int64_t i = col_start * vec_nnz; i < col_end * vec_nnz; ++i) {
        int64_t row = nzidx2row[i];
        if (row_start <= row && row < row_end)  {
            rows_view[nnz] = row - row_start;
            cols_view[nnz] = nzidx2col[i] - col_start;
            vals_view[nnz] = nzidx2val[i];
            nnz += 1;
        }
    }
    return nnz;
}

template <typename T>
static void apply_cscoo_submat_to_vector_from_left(
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Sv,          // Sv += S * v.
    int64_t incSv,  // stride between elements of Sv
    const int64_t *rows_view,
    const int64_t *cols_view,
    const T       *vals_view,
    int64_t num_cols,
    int64_t nnz
) {
    // (rows_view, cols_view, vals_view) define a sparse matrix
    // "S" in COO-format with a CSC-like property.
    //
    //      The CSC-like property requires that all nonzero entries
    //      in a given column of the sparse matrix are contiguous in
    //      "cols_view".
    //  
    //      The sparse matrix does not need to be wide.
    //
    int64_t i = 0;
    for (int64_t c = 0; c < num_cols; ++c) {
        T scale = v[c * incv];
        while (cols_view[i] == c && i < nnz) {
            int64_t row = rows_view[i];
            Sv[row * incSv] += (vals_view[i] * scale);
            i += 1;
        }
    }
}

template <typename T>
static void apply_cscoo_left(
    blas::Layout layout,
    int64_t d,
    int64_t n,
    int64_t m,
    SparseSkOp<T>& S0,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T *B,
    int64_t ldb,
    int threads
) {
    int64_t vec_nnz = S0.dist.vec_nnz;
    int64_t *S_rows = new int64_t[m * vec_nnz]{};
    int64_t *S_cols = new int64_t[m * vec_nnz]{};
    T       *S_vals = new       T[m * vec_nnz]{};
    int64_t nnz = filter_regular_cscoo<T>(
        S0.rows, S0.cols, S0.vals, vec_nnz,
        j_os, j_os + m,
        i_os, i_os + d,
        S_rows, S_cols, S_vals
    );
    // The implementation of filter_regular_cscoo has a HARD requirement
    // that S0 has a fixed number of nonzeros per column.
    //
    // Once we have (S_rows, S_cols, S_vals) in the format ensured
    // by filter_regular_cscoo, we apply the resulting sparse matrix "S"
    // to the left of A to get B = S*A.
    //
    // This function does not require that S or S0 is wide.

    omp_set_num_threads(threads);
    #pragma omp parallel default(shared)
    {
        // Setup variables for the current thread
        const T *A_col = nullptr;
        T *B_col = nullptr;
        // Do the work for the current thread.
        if (layout == blas::Layout::ColMajor) {
            #pragma omp for schedule(static)
            for (int64_t k = 0; k < n; k++) {
                A_col = &A[lda * k];
                B_col = &B[ldb * k];
                apply_cscoo_submat_to_vector_from_left<T>(
                    A_col, 1, B_col, 1,
                    S_rows, S_cols, S_vals, m, nnz
                );
            }
        } else {
            #pragma omp for schedule(static)
            for (int64_t k = 0; k < n; k++) {
                A_col = &A[k];
                B_col = &B[k];
                apply_cscoo_submat_to_vector_from_left<T>(
                    A_col, lda, B_col, ldb,
                    S_rows, S_cols, S_vals, m, nnz
                );
            }
        }
    }

    delete [] S_rows;
    delete [] S_cols;
    delete [] S_vals;
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
    int64_t ldb,
    int threads // default is 4.
) {
    randblas_require(S0.rows != NULL); // must be filled.
    randblas_require(fixed_nnz_per_col(S0));
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
        randblas_require(false); // not implemented.
        // rows_S = m;
        // cols_S = d;
    }
    // ^ Implementation limitation.
    // Dimensionality sanity checks, and perform the sketch.
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }
    apply_cscoo_left<T>(layout, d, n, m, S0, i_os, j_os, A, lda, B, ldb, threads);
    return;
}

template RNGState fill_sparse<float>(SparseSkOp<float> &S0);
template RNGState fill_sparse<double>(SparseSkOp<double> &S0);

template void print_saso<float>(SparseSkOp<float> &S0);
template void print_saso<double>(SparseSkOp<double> &S0);

template void lskges<float>(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, float alpha,
    SparseSkOp<float> &S0, int64_t i_os, int64_t j_os, const float *A, int64_t lda, float beta, float *B, int64_t ldb, int threads);
template void lskges<double>(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, double alpha,
    SparseSkOp<double> &S0, int64_t i_os, int64_t j_os, const double *A, int64_t lda, double beta, double *B, int64_t ldb, int threads);


} // end namespace RandBLAS::sparse_ops
