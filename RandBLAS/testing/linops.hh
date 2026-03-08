// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/skge.hh"
#include "RandBLAS/sparse_data/spmm_dispatch.hh"
#include "RandBLAS/util.hh"
#include <functional>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>


namespace RandBLAS::testing {

using blas::Layout;
using blas::Op;
using RandBLAS::sparse_data::COOMatrix;
using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::CSCMatrix;

#ifdef __cpp_concepts
using RandBLAS::SparseMatrix;
#else
#define SparseMatrix typename
#endif
using RandBLAS::SparseSkOp;
using RandBLAS::DenseSkOp;
using RandBLAS::RNGState;
using RandBLAS::DenseDist;
using RandBLAS::dims_before_op;
using RandBLAS::offset_and_ldim;
using RandBLAS::layout_to_strides;
using RandBLAS::dims64_t;


template <typename T>
std::vector<T> eye(int64_t n) {
    std::vector<T> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        A[i + n*i] = 1.0;
    return A;
}

template <typename T, typename RNG=r123::Philox4x32>
auto random_matrix(int64_t m, int64_t n, RNGState<RNG> s) {
    std::vector<T> A(m * n);
    DenseDist DA(m, n);
    auto next_state = RandBLAS::fill_dense(DA, A.data(), s);
    std::tuple<std::vector<T>, Layout, RNGState<RNG>> t{A, DA.natural_layout, next_state};
    return t;
}

template <typename LINOP>
dims64_t dimensions(LINOP &S) {return {S.n_rows, S.n_cols};}

template <typename T, SparseMatrix SpMat>
void to_explicit_buffer(const SpMat &a, T *mat_a, Layout layout) {
    using sint_t = typename SpMat::index_t;
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    if constexpr (is_coo) {
        RandBLAS::sparse_data::coo::coo_to_dense(a, layout, mat_a);
    } else if constexpr (is_csc) {
        RandBLAS::sparse_data::csc::csc_to_dense(a, layout, mat_a);
    } else if constexpr (is_csr) {
        RandBLAS::sparse_data::csr::csr_to_dense(a, layout, mat_a);
    } else {
        randblas_require(false);
    }
    return;
}

template <typename T>
void to_explicit_buffer(const SparseSkOp<T> &a, T *mat_a, Layout layout) {
    if (a.nnz < 0) {
        SparseSkOp<T> a_shallowcopy(a.dist, a.seed_state);
        fill_sparse(a_shallowcopy);
        auto a_coo = RandBLAS::sparse::coo_view_of_skop(a_shallowcopy);
        to_explicit_buffer(a_coo, mat_a, layout);
    } else {
        auto a_coo = RandBLAS::sparse::coo_view_of_skop(a);
        to_explicit_buffer(a_coo, mat_a, layout);
    }
    return;
}

template <typename T>
void to_explicit_buffer(const DenseSkOp<T> &s, T *mat_s, Layout layout) {
    auto n_rows = s.dist.n_rows;
    auto n_cols = s.dist.n_cols;
    auto [stride_row, stride_col] = layout_to_strides(layout, n_rows, n_cols);
    #define MAT_S(_i, _j) mat_s[(_i) * stride_row + (_j) * stride_col ]

    // for some reason we prefer to make a copy rather than pass-by-value.
    DenseSkOp<T> s_copy(s.dist, s.seed_state);
    RandBLAS::fill_dense(s_copy);
    auto [buff_stride_row, buff_stride_col] = layout_to_strides(s_copy.layout, n_rows, n_cols);
    #define BUFF(_i, _j) s_copy.buff[(_i) * buff_stride_row + (_j) * buff_stride_col]

    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            MAT_S(i, j) = BUFF(i, j);
        }
    }
    return;
}


// MARK:      Multiply from the LEFT
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


template <typename T>
void left_apply(Layout layout, Op opS, Op opA, int64_t d, int64_t n, int64_t m, T alpha, SparseSkOp<T> &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb, int threads = 0) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse::lskges(layout, opS, opA, d, n, m, alpha, S, S_ro, S_co, A, lda, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
    return;
}

template <typename T>
void left_apply(Layout layout, Op opS, Op opA, int64_t d, int64_t n, int64_t m, T alpha, DenseSkOp<T> &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb, int threads = 0) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::dense::lskge3(layout, opS, opA, d, n, m, alpha, S, S_ro, S_co, A, lda, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
    return;
}

template <typename T, SparseMatrix SpMat>
void left_apply(Layout layout, Op opS, Op opA, int64_t d, int64_t n, int64_t m, T alpha, const SpMat &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, int64_t ldb, int threads = 0) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse_data::left_spmm(layout, opS, opA, d, n, m, alpha, S, S_ro, S_co, A, lda, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
    return;
}

template <typename T, typename LinOp>
void reference_left_apply(
    Layout layout, Op transS, Op transA, int64_t d, int64_t n, int64_t m, T alpha, LinOp &S, int64_t S_ro, int64_t S_co, const T *A, int64_t lda, T beta, T *B, T *E, int64_t ldb
){
    randblas_require(d > 0);
    randblas_require(m > 0);
    randblas_require(n > 0);

    // Dimensions of mat(A), rather than op(mat(A))
    auto [rows_S, cols_S] = dimensions(S);
    auto [rows_mat_A   , cols_mat_A   ] = RandBLAS::dims_before_op(m, n, transA);
    auto [rows_submat_S, cols_submat_S] = RandBLAS::dims_before_op(d, m, transS);

    // Sanity checks on dimensions and strides
    int64_t lds, s_row_stride, s_col_stride, pos, size_A, size_B;
    if (layout == Layout::ColMajor) {
        lds = rows_S;
        pos = S_ro + lds * S_co;
        randblas_require(lds >= rows_submat_S);
        randblas_require(lda >= rows_mat_A);
        randblas_require(ldb >= d);
        size_A = lda * (cols_mat_A - 1) + rows_mat_A;;
        size_B = ldb * (n - 1) + d;
        s_row_stride = 1;
        s_col_stride = lds;
    } else {
        lds = cols_S;
        pos = S_ro * lds + S_co;
        randblas_require(lds >= cols_submat_S);
        randblas_require(lda >= cols_mat_A);
        randblas_require(ldb >= n);
        size_A = lda * (rows_mat_A - 1) + cols_mat_A;
        size_B = ldb * (d - 1) + n;
        s_row_stride = lds;
        s_col_stride = 1;
    }

    auto size_S = rows_S * cols_S;
    std::vector<T> S_dense(size_S);
    std::vector<T> S_dense_abs(size_S);
    to_explicit_buffer(S, S_dense.data(), layout);
    for (int64_t i = 0; i < rows_S; ++i) {
        for (int64_t j = 0; j < cols_S; ++j) {
            auto ell = i * s_row_stride + j * s_col_stride;
            S_dense_abs[ell] = abs(S_dense[ell]);
        }
    }

    // Compute the reference value
    T* S_ptr = S_dense.data();
    blas::gemm(layout, transS, transA, d, n, m,
        alpha, &S_ptr[pos], lds, A, lda, beta, B, ldb
    );

    // Compute the matrix needed for componentwise error bounds.
    std::vector<T> A_abs_vec(size_A);
    T* A_abs = A_abs_vec.data();
    for (int64_t i = 0; i < size_A; ++i)
        A_abs[i] = abs(A[i]);
    if (beta != 0.0) {
        for (int64_t i = 0; i < size_B; ++i)
            E[i] = abs(B[i]);
    }
    T eps = std::numeric_limits<T>::epsilon();
    T err_alpha = (abs(alpha) * m) * (2 * eps);
    T err_beta = abs(beta) * eps;
    T* S_abs_ptr = S_dense_abs.data();
    blas::gemm(layout, transS, transA, d, n, m,
        err_alpha, &S_abs_ptr[pos], lds, A_abs, lda, err_beta, E, ldb
    );
    return;
}


// MARK:      Multiply from the RIGHT
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <typename T>
void right_apply(Layout layout, Op transA, Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, const SparseSkOp<T> &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb, int threads) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse::rskges(layout, transA, transS, m, d, n, alpha, A, lda, S, S_ro, S_co, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
}

template <typename T>
void right_apply(Layout layout, Op transA, Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, const DenseSkOp<T> &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb, int threads) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::dense::rskge3(layout, transA, transS, m, d, n, alpha, A, lda, S, S_ro, S_co, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
}

template <typename T, SparseMatrix SpMat>
void right_apply(Layout layout, Op transA, Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, const SpMat &S, int64_t S_ro, int64_t S_co, T beta, T *B, int64_t ldb, int threads) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse_data::right_spmm(layout, transA, transS, m, d, n, alpha, A, lda, S, S_ro, S_co, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
}

template <typename T, typename LinOp>
void reference_right_apply(
    Layout layout, Op transA, Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, const LinOp &S0, int64_t S_ro, int64_t S_co, T beta, T *B, T *E, int64_t ldb
) {
    // Check dimensions of submat(S).
    auto [submat_S_rows, submat_S_cols] = RandBLAS::dims_before_op(n, d, transS);
    auto [rows_S, cols_S] = dimensions(S0);
    randblas_require(submat_S_rows <= rows_S);
    randblas_require(submat_S_cols <= cols_S);
    // Check dimensions of mat(A).
    auto [mat_A_rows, mat_A_cols] = RandBLAS::dims_before_op(m, n, transA);
    if (layout == Layout::ColMajor) {
        randblas_require(lda >= mat_A_rows);
    } else {
        randblas_require(lda >= mat_A_cols);
    }
    //
    // Compute B = op(A) op(submat(S)) by left_apply. We start with the identity
    //
    //      B^T = op(submat(S))^T op(A)^T
    //
    // Then we interchange the operator "op" for op(A) and the operator (*)^T.
    //
    //      B^T = op(submat(S))^T op(A^T)
    //
    // We tell left_apply to process (B^T) and (A^T) in the opposite memory layout
    // compared to the layout for (A, B).
    //
    auto trans_transS = (transS == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    reference_left_apply(
        trans_layout, trans_transS, transA,
        d, m, n, alpha, S0, S_ro, S_co, A, lda, beta, B, E, ldb
    );
}

} // end namespace RandBLAS::testing
