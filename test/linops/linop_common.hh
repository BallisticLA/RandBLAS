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

#include "RandBLAS/testing/linops.hh"
#include "RandBLAS/testing/comparison.hh"

#include <gtest/gtest.h>


namespace test::linop_common {

using blas::Layout;
using blas::Op;
using RandBLAS::RNGState;
using RandBLAS::offset_and_ldim;

using RandBLAS::testing::reference_left_apply;
using RandBLAS::testing::left_apply;
using RandBLAS::testing::reference_right_apply;
using RandBLAS::testing::right_apply;

using RandBLAS::testing::to_explicit_buffer;
using RandBLAS::testing::dimensions;

using RandBLAS::testing::random_matrix;
using RandBLAS::testing::eye;


template <typename T, typename LinOp>
void test_left_apply_to_random(
    // B = alpha * S * A + beta*B, where A is m-by-n and random, S is m-by-d, and B is d-by-n and random
    T alpha, LinOp &S, int64_t n, T beta, Layout layout, int threads = 0
) {
    auto [d, m] = dimensions(S);
    auto A  = std::get<0>(random_matrix<T>(m, n, RandBLAS::RNGState(99)));
    auto B0 = std::get<0>(random_matrix<T>(d, n, RandBLAS::RNGState(42)));
    std::vector<T> B1(B0);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t lda = (is_colmajor) ? m : n;
    int64_t ldb = (is_colmajor) ? d : n;

    // compute S*A.
    left_apply<T>(
        layout, Op::NoTrans, Op::NoTrans,
        d, n, m,
        alpha, S, 0, 0, A.data(), lda,
        beta, B0.data(), ldb, threads
    );

    // compute expected result (B1) and allowable error (E)
    std::vector<T> E(d * n, 0.0);
    reference_left_apply<T>(
        layout, Op::NoTrans, Op::NoTrans,
        d, n, m,
        alpha, S, 0, 0, A.data(), lda,
        beta, B1.data(), E.data(), ldb
    );

    // check the result
    auto msg = RandBLAS::testing::buffs_approx_equal<T>(
        B0.data(), B1.data(), E.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

    return;
}

template <typename T, typename LinOp>
void test_left_apply_submatrix_to_eye(
    // B = alpha * submat(S0) * eye + beta*B, where S = submat(S) is d1-by-m1 offset by (S_ro, S_co) in S0, and B is random.
    T alpha, LinOp &S0, int64_t d1, int64_t m1, int64_t S_ro, int64_t S_co, Layout layout, T beta = 0.0, int threads = 0
) {
    auto [d0, m0] = dimensions(S0);
    randblas_require(d0 >= d1);
    randblas_require(m0 >= m1);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t lda = m1;
    int64_t ldb = (is_colmajor) ? d1 : m1;

    // define a matrix to be sketched, and create workspace for sketch.
    auto A = eye<T>(m1);
    auto B = std::get<0>(random_matrix<T>(d1, m1, RandBLAS::RNGState(42)));
    std::vector<T> B_backup(B);

    // Perform the sketch
    left_apply(
        layout, Op::NoTrans, Op::NoTrans,
        d1, m1, m1,
        alpha, S0, S_ro, S_co,
        A.data(), lda,
        beta, B.data(), ldb, threads
    );

    // Check the result
    T *expect = new T[d0 * m0];
    to_explicit_buffer(S0, expect, layout);
    int64_t ld_expect = (is_colmajor) ? d0 : m0;
    auto [row_stride_s, col_stride_s] = RandBLAS::layout_to_strides(layout, ld_expect);
    auto [row_stride_b, col_stride_b] = RandBLAS::layout_to_strides(layout, ldb);
    int64_t offset = row_stride_s * S_ro + col_stride_s * S_co;
    #define MAT_E(_i, _j) expect[offset + (_i)*row_stride_s + (_j)*col_stride_s]
    #define MAT_B(_i, _j) B_backup[       (_i)*row_stride_b + (_j)*col_stride_b]
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < m1; ++j) {
            MAT_E(i,j) = alpha * MAT_E(i,j) + beta * MAT_B(i, j);
        }
    }

    auto msg = RandBLAS::testing::matrices_approx_equal(
        layout, Op::NoTrans,
        d1, m1,
        B.data(), ldb,
        &expect[offset], ld_expect,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

    delete [] expect;
}

template <typename T, typename LinOp>
void test_left_apply_transpose_to_eye(
    // B = S^T * eye, where S is m-by-d, B is d-by-m
    LinOp &S, Layout layout, int threads = 0
) {
    auto [m, d] = dimensions(S);
    auto A = eye<T>(m);
    std::vector<T> B(d * m, 0.0);
    bool is_colmajor = (Layout::ColMajor == layout);
    int64_t ldb = (is_colmajor) ? d : m;
    int64_t lds = (is_colmajor) ? m : d;

    left_apply<T>(
        layout,
        Op::Trans,
        Op::NoTrans,
        d, m, m,
        1.0, S, 0, 0, A.data(), m,
        0.0, B.data(), ldb, threads
    );

    std::vector<T> S_dense(m * d, 0.0);
    to_explicit_buffer(S, S_dense.data(), layout);
    auto msg = RandBLAS::testing::matrices_approx_equal(
        layout, Op::Trans, d, m,
        B.data(), ldb, S_dense.data(), lds,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }
}

template <typename T, typename LinOp>
void test_left_apply_to_submatrix(
    // B = S * A, where S is d-by-m, A = A0[A_ro:(A_ro + m), A_co:(A_co + n)], and A0 is random m0-by-n0.
    LinOp &S, int64_t n, int64_t m0, int64_t n0, int64_t A_ro, int64_t A_co, Layout layout, int threads = 0
) {
    auto [d, m] = dimensions(S);
    randblas_require(m0 >= m);
    randblas_require(n0 >= n);

    std::vector<T> B0(d * n, 0.0);
    int64_t ldb = (layout == Layout::ColMajor) ? d : n;

    auto A = std::get<0>(random_matrix<T>(m0, n0, RNGState(13)));
    auto [a_offset, lda] = offset_and_ldim(layout, m0, n0, A_ro, A_co);
    T *A_ptr = &A.data()[a_offset];
    left_apply<T>(
        layout,
        Op::NoTrans,
        Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        A_ptr, lda,
        0.0, B0.data(), ldb, threads
    );

    std::vector<T> B1(d * n, 0.0);
    std::vector<T> E(d * n, 0.0);
    reference_left_apply<T>(
        layout,
        Op::NoTrans,
        Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        A_ptr, lda,
        0.0, B1.data(), E.data(), ldb
    );
    auto msg = RandBLAS::testing::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

}

template <typename T, typename LinOp>
void test_left_apply_to_transposed(
    // B = S * A^T, where S is d-by-m, A is m-by-n and random
    LinOp &S, int64_t n, Layout layout, int threads = 0
) {
    auto [d, m] = dimensions(S);
    auto At = std::get<0>(random_matrix<T>(n, m, RNGState(101)));
    std::vector<T> B0(d * n, 0.0);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t lda = (is_colmajor) ? n : m;
    int64_t ldb = (is_colmajor) ? d : n;

    left_apply<T>(
        layout,
        Op::NoTrans,
        Op::Trans,
        d, n, m,
        1.0, S, 0, 0,
        At.data(), lda,
        0.0, B0.data(), ldb, threads
    );

    std::vector<T> B1(d * n, 0.0);
    std::vector<T> E(d * n, 0.0);
    reference_left_apply<T>(
        layout, Op::NoTrans, Op::Trans,
        d, n, m,
        1.0, S, 0, 0,
        At.data(), lda,
        0.0, B1.data(), E.data(), ldb
    );
    auto msg = RandBLAS::testing::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

}


// MARK:      Multiply from the RIGHT
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <typename T, typename LinOp>
void test_right_apply_to_random(
    // B = alpha * A * S + beta * B, where A is m-by-n, S is n-by-d, B is m-by-d and random
    T alpha, LinOp &S, int64_t m, Layout layout, T beta, int threads = 0
) {
    auto [n, d] = dimensions(S);
    auto A  = std::get<0>(random_matrix<T>(m, n, RandBLAS::RNGState(57)));
    auto B0 = std::get<0>(random_matrix<T>(m, d, RandBLAS::RNGState(10)));
    std::vector<T> B1(B0);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t lda = (is_colmajor) ? m : n;
    int64_t ldb = (is_colmajor) ? m : d;

    right_apply<T>(
        layout, Op::NoTrans, Op::NoTrans,
        m, d, n, alpha, A.data(), lda, S, 0, 0,
        beta, B0.data(), ldb, threads
    );

    std::vector<T> E(m * d, 0.0);
    reference_right_apply(
        layout, Op::NoTrans, Op::NoTrans,
        m, d, n, alpha, A.data(), lda, S, 0, 0,
        beta, B1.data(), E.data(), ldb
    );

    auto msg = RandBLAS::testing::buffs_approx_equal<T>(
        B0.data(), B1.data(), E.data(), m * d,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }
}

template <typename T, typename LinOp>
void test_right_apply_submatrix_to_eye(
    // B = alpha * eye * submat(S) + beta*B : submat(S) is n-by-d, eye is n-by-n, B is n-by-d and random
    T alpha, LinOp &S0, int64_t n, int64_t d, int64_t S_ro, int64_t S_co, Layout layout, T beta = 0.0, int threads = 0
) {
    auto [n0, d0] = dimensions(S0);
    randblas_require(n0 >= n);
    randblas_require(d0 >= d);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t lda = n;
    int64_t ldb = (is_colmajor) ? n : d;

    auto A = eye<T>(n);
    auto B = std::get<0>(random_matrix<T>(n, d, RandBLAS::RNGState(11)));
    std::vector<T> B_backup(B);
    right_apply(layout, Op::NoTrans, Op::NoTrans, n, d, n, alpha, A.data(), lda, S0, S_ro, S_co, beta, B.data(), ldb, threads);

    T *expect = new T[n0 * d0];
    to_explicit_buffer(S0, expect, layout);
    int64_t ld_expect = (is_colmajor)? n0 : d0;
    auto [row_stride_s, col_stride_s] = RandBLAS::layout_to_strides(layout, ld_expect);
    auto [row_stride_b, col_stride_b] = RandBLAS::layout_to_strides(layout, ldb);
    int64_t offset = row_stride_s * S_ro + col_stride_s * S_co;
    #define MAT_E(_i, _j) expect[offset + (_i)*row_stride_s + (_j)*col_stride_s]
    #define MAT_B(_i, _j) B_backup[       (_i)*row_stride_b + (_j)*col_stride_b]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            MAT_E(i,j) = alpha * MAT_E(i,j) + beta * MAT_B(i, j);
        }
    }

    auto msg = RandBLAS::testing::matrices_approx_equal(
        layout, Op::NoTrans, n, d, B.data(), ldb, &expect[offset], ld_expect,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

    delete [] expect;
}

template <typename T, typename LinOp>
void test_right_apply_transpose_to_eye(
    // B = eye * S^T, where S is d-by-n, so eye is order n and B is n-by-d
    LinOp &S, Layout layout, int threads = 0
) {
    auto [d, n] = dimensions(S);
    auto A = eye<T>(n);
    std::vector<T> B(n * d, 0.0);
    bool is_colmajor = Layout::ColMajor == layout;
    int64_t ldb = (is_colmajor) ? n : d;
    int64_t lds = (is_colmajor) ? d : n;

    right_apply<T>(layout, Op::NoTrans, Op::Trans, n, d, n, 1.0, A.data(), n, S, 0, 0, 0.0, B.data(), ldb, threads);

    std::vector<T> S_dense(n * d, 0.0);
    to_explicit_buffer(S, S_dense.data(), layout);
    auto msg = RandBLAS::testing::matrices_approx_equal(
        layout, Op::Trans, n, d,
        B.data(), ldb, S_dense.data(), lds,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }
}

template <typename T, typename LinOp>
void test_right_apply_to_submatrix(
    // B = submat(A) * S, where mat(A) is m0-by-n0, S is n-by-d, and submat(A) is m-by-n, B is m-by-d
    LinOp &S, int64_t m, int64_t m0, int64_t n0, int64_t A_ro, int64_t A_co, Layout layout, int threads = 0
) {
    auto [n, d] = dimensions(S);
    randblas_require(m0 >= m);
    randblas_require(n0 >= n);

    std::vector<T> B0(m * d, 0.0);
    int64_t ldb = (layout == Layout::ColMajor) ? m : d;

    auto A = std::get<0>(random_matrix<T>(m0, n0, RandBLAS::RNGState(1)));
    auto [a_offset, lda] = offset_and_ldim(layout, m0, n0, A_ro, A_co);
    T *A_ptr = &A.data()[a_offset];

    right_apply<T>(layout, Op::NoTrans, Op::NoTrans, m, d, n, 1.0, A_ptr, lda, S, 0, 0, 0.0, B0.data(), ldb, threads);

    std::vector<T> B1(d * m, 0.0);
    std::vector<T> E(d * m, 0.0);
    reference_right_apply<T>(
        layout,
        Op::NoTrans,
        Op::NoTrans,
        m, d, n,
        1.0, A_ptr, lda, S, 0, 0,
        0.0, B1.data(), E.data(), ldb
    );
    auto msg = RandBLAS::testing::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), d * m,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

}

template <typename T, typename LinOp>
void test_right_apply_to_transposed(
    // B = A^T S, where A is n-by-m, S is n-by-d, and B is m-by-d
    LinOp &S, int64_t m, Layout layout, int threads = 0
) {
    auto [n, d] = dimensions(S);
    auto At = std::get<0>(random_matrix<T>(n, m, RNGState(0)));
    std::vector<T> B0(m * d, 0.0);
    bool is_colmajor = (layout == Layout::ColMajor);
    int64_t lda = (is_colmajor) ? n : m;
    int64_t ldb = (is_colmajor) ? m : d;

    right_apply<T>(layout, Op::Trans, Op::NoTrans, m, d, n, 1.0, At.data(), lda, S, 0, 0, 0.0, B0.data(), ldb, threads);

    std::vector<T> B1(m * d, 0.0);
    std::vector<T> E(m * d, 0.0);
    reference_right_apply<T>(
        layout, Op::Trans, Op::NoTrans,
        m, d, n,
        1.0, At.data(), lda, S, 0, 0,
        0.0, B1.data(), E.data(), ldb
    );
    auto msg = RandBLAS::testing::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), m * d,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }

}

} // end namespace test::linop_common
