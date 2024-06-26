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

#include "test/test_datastructures/test_spmats/common.hh"
#include "test/test_matmul_impls/linop_common.hh"
#include <gtest/gtest.h>

using namespace test::linop_common;
using blas::Layout;

using test::test_datastructures::test_spmats::iid_sparsify_random_dense;
// ^ Not used in this file, but required by client files.


template <SparseMatrix SpMat>
class TestLeftMultiply_Sparse : public ::testing::Test {
    // C = alpha * opA(submat(A)) @ opB(B) + beta * C
    //
    // In what follows, "self" refers to A and "other" refers to B.
    //

    using T = typename SpMat::scalar_t;
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    virtual SpMat make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) = 0;

    void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0);
    }

    void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        randblas_require(alpha != (T)1.0 || beta != (T)0.0);
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta);
    }

    void transpose_self(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_transpose_to_eye<T>(A, layout);
    }

    void submatrix_self(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // rows in A
        int64_t m,      // columns in A, rows in B = eye(m)
        int64_t d0,     // rows in A0
        int64_t m0,     // cols in A0
        int64_t A_ro,   // row offset for A in A0
        int64_t A_co,   // column offset for A in A0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        randblas_require(d0 > d);
        randblas_require(m0 > m);
        auto A0 = this->make_test_matrix(d0, m0, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A0, d, m, A_ro, A_co, layout, 0.0);
    }

    void submatrix_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        int64_t m0,    // rows in B0
        int64_t n0,    // cols in B0
        int64_t B_ro,  // row offset for B in B0
        int64_t B_co,  // column offset for B in B0
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(d, m, p, key);
        randblas_require(m0 > m);
        randblas_require(n0 > n);
        test_left_apply_to_submatrix<T>(A, n, m0, n0, B_ro, B_co, layout);
    }

    void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(d, m, p, key);
        test_left_apply_to_transposed<T>(A, n, layout);
    }

};


template <SparseMatrix SpMat>
class TestRightMultiply_Sparse : public ::testing::Test {
    // C = alpha * opB(B) @ opA(submat(A)) + beta * C
    //
    //  In what follows, "self" refers to A and "other" refers to B.
    //

    using T = typename SpMat::scalar_t;
    protected:
    virtual void SetUp(){};
    virtual void TearDown(){};

    virtual SpMat make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) = 0;

    void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_right_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0, 0);
    }

    void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
       test_right_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta, 0);
    }

    void transpose_self(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_right_apply_tranpose_to_eye<T>(A, layout, 0);
    }

    void submatrix_self(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // cols in A
        int64_t m,      // rows in A, columns in B = eye(m)
        int64_t d0,     // cols in A0
        int64_t m0,     // rows in A0
        int64_t A_ro,   // row offset for A in A0
        int64_t A_co,   // col offset for A in A0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        auto A0 = this->make_test_matrix(m0, d0, p, key);
        test_right_apply_submatrix_to_eye<T>(1.0, A0, m, d, A_ro, A_co, layout, 0.0, 0);
    }

    void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // cols in A
        int64_t n,     // rows in A and B
        int64_t m,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(n, d, p, key);
        test_right_apply_to_transposed<T>(A, m, layout, 0);
    }

    void submatrix_other(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // cols in A
        int64_t m,      // rows in B
        int64_t n,      // rows in A, columns in B
        int64_t m0,     // rows in B0
        int64_t n0,     // cols in B0
        int64_t B_ro,   // row offset for B in B0
        int64_t B_co,   // col offset for B in B0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(n, d, p, key);
        test_right_apply_to_submatrix<T>(A, m, m0, n0, B_ro, B_co, layout, 0);
    }
};
