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
    int64_t *la_idxs, *sa_idxs;

    if (S0.dist.n_rows <= S0.dist.n_cols) {
        la_idxs = S0.cols; // indices of nonzeros for the long-axis
        sa_idxs = S0.rows; // indices of nonzeros for the short-axis
    } else {
        sa_idxs = S0.cols; // indices of nonzeros for the short-axis
        la_idxs = S0.rows; // indices of nonzeros for the long-axis
    }

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

template RNGState template_fill_saso<float, Philox>(SparseSkOp<float> &S0);
template RNGState template_fill_saso<double, Philox>(SparseSkOp<double> &S0);
template RNGState template_fill_saso<float, Threefry>(SparseSkOp<float> &S0);
template RNGState template_fill_saso<double, Threefry>(SparseSkOp<double> &S0);

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
static int64_t filter_cscoo(
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
static void cscoo_matvec(
    const T *v,
    int64_t incv, // stride between elements of v
    T *Sv, // Sv += S0[:, col_start:col_end] * v.
    int64_t incSv, // stride between elements of Sv
    const int64_t *rows_view,
    const int64_t *cols_view,
    const T       *vals_view,
    int64_t num_cols,
    int64_t nnz
) {
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
static void apply_wide_saso_left(
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
    RandBLAS::sparse::SparseDist D = S0.dist;
    int64_t r0 = i_os;
    int64_t c0 = j_os;
    int64_t rf = r0 + d;
    int64_t cf = c0 + m;

    int64_t vec_nnz = D.vec_nnz;
    int64_t *rows_view = new int64_t[m * vec_nnz]{};
    int64_t *cols_view = new int64_t[m * vec_nnz]{};
    T       *vals_view = new       T[m * vec_nnz]{};
    int64_t nnz = filter_cscoo<T>(
        S0.rows, S0.cols, S0.vals, vec_nnz, c0, cf, r0, rf,
        rows_view, cols_view, vals_view
    );

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
                cscoo_matvec<T>(A_col, 1, B_col, 1, rows_view, cols_view, vals_view, m, nnz);
            }
        } else {
            #pragma omp for schedule(static)
            for (int64_t k = 0; k < n; k++) {
                A_col = &A[k];
                B_col = &B[k];
                cscoo_matvec<T>(A_col, lda, B_col, ldb, rows_view, cols_view, vals_view, m, nnz);
            }
        }
    }

    delete [] rows_view;
    delete [] cols_view;
    delete [] vals_view;
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
    // randblas_require(d <= m);
    //  ^ Sketching can't increase dimension, but sometimes we need to "lift" something that's
    //    been sketched back to the original (higher) dimension.
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
        randblas_require(false); // not implemented. The only reasonable next step implementation-wise is below.
        // rows_S = m;
        // cols_S = d;
        // if (rows_S < cols_S && S0.dist.family == SparseDistName::SASO) {
        //     // This dimensionality check is just to make sure the transpose of the SASO has a fixed number
        //     // of nonzeros per column. (It's possible that we want to take linear combinations of columns
        //     // of a tall SASO or rows of a wide SASO!)
        //     throw std::runtime_error(std::string("Not implemented. We need op(S) to have a fixed number
        //     of nonzeros per column."))
        // }
        // ^ Implementation-wise, could have a function that returns a wide-SASO view of a transpose of a tall SASO.
    }
    randblas_require(S0.dist.family == SparseDistName::SASO);
    // ^ Implementation limitation.
    // Dimensionality sanity checks, and perform the sketch.
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }
    apply_wide_saso_left<T>(layout, d, n, m, S0, i_os, j_os, A, lda, B, ldb, threads);
    return;
}

template RNGState fill_saso<float>(SparseSkOp<float> &S0);
template RNGState fill_saso<double>(SparseSkOp<double> &S0);

template void print_saso<float>(SparseSkOp<float> &S0);
template void print_saso<double>(SparseSkOp<double> &S0);

template void lskges<float>(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, float alpha,
    SparseSkOp<float> &S0, int64_t i_os, int64_t j_os, const float *A, int64_t lda, float beta, float *B, int64_t ldb, int threads);
template void lskges<double>(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, double alpha,
    SparseSkOp<double> &S0, int64_t i_os, int64_t j_os, const double *A, int64_t lda, double beta, double *B, int64_t ldb, int threads);


} // end namespace RandBLAS::sparse_ops
