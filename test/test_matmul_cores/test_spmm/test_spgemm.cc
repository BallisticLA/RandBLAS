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

#include "RandBLAS/config.h"

#if defined(RandBLAS_HAS_MKL)

#include "test/test_datastructures/test_spmats/common.hh"
#include "test/test_matmul_cores/linop_common.hh"
#include "test/comparison.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;
using namespace RandBLAS::sparse_data::coo;
using blas::Layout;
using blas::Op;

using test::test_datastructures::test_spmats::iid_sparsify_random_dense;
using test::linop_common::to_explicit_buffer;
using test::linop_common::random_matrix;


template <typename T>
class TestSpGEMM : public ::testing::Test {
    // Tests for RandBLAS::spgemm: C = alpha * op(A) * B + beta * C
    //   where A and B are sparse, C is dense.

    protected:

    virtual void SetUp(){};
    virtual void TearDown(){};

    CSRMatrix<T> make_csr(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        CSRMatrix<T> A(m, n);
        std::vector<T> buf(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, buf.data(), 1 - nonzero_prob, s);
        dense_to_csr<T>(Layout::ColMajor, buf.data(), 0.0, A);
        return A;
    }

    COOMatrix<T> make_coo(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        COOMatrix<T> A(m, n);
        std::vector<T> buf(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, buf.data(), 1 - nonzero_prob, s);
        dense_to_coo<T>(Layout::ColMajor, buf.data(), 0.0, A);
        return A;
    }

    // Core test: C = alpha * op(A) * B + beta * C
    //   A is m-by-k (before op), B is k-by-n, C is m-by-n.
    //   Reference: densify both, use blas::gemm.
    template <SparseMatrix SpMat1, SparseMatrix SpMat2>
    void test_spgemm(
        Op opA,
        int64_t m, int64_t n, int64_t k,
        T alpha, T beta,
        const SpMat1 &A, const SpMat2 &B,
        Layout layout
    ) {
        bool is_colmajor = (layout == Layout::ColMajor);
        int64_t ldc = (is_colmajor) ? m : n;

        // Compute reference: densify A and B, then gemm.
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, opA);
        std::vector<T> A_dense(rows_A * cols_A, 0.0);
        std::vector<T> B_dense(k * n, 0.0);
        to_explicit_buffer<T>(A, A_dense.data(), layout);
        to_explicit_buffer<T>(B, B_dense.data(), layout);

        int64_t lda = (is_colmajor) ? rows_A : cols_A;
        int64_t ldb = (is_colmajor) ? k : n;

        // Generate random initial C (for beta != 0 tests).
        auto C_ref = std::get<0>(random_matrix<T>(m, n, RandBLAS::RNGState(42)));
        std::vector<T> C_actual(C_ref);

        // Reference: blas::gemm
        blas::gemm(layout, opA, Op::NoTrans, m, n, k,
            alpha, A_dense.data(), lda, B_dense.data(), ldb,
            beta, C_ref.data(), ldc);

        // Actual: RandBLAS::spgemm
        RandBLAS::spgemm(layout, opA, m, n, k,
            alpha, A, B, beta, C_actual.data(), ldc);

        // Compare with componentwise error bounds.
        // Error model: |C_actual - C_ref| <= |alpha| * k * 2*eps * |A_dense| * |B_dense| + |beta| * eps * |C_orig|
        // We compute the error bound via gemm on absolute values.
        T eps = std::numeric_limits<T>::epsilon();
        T err_alpha = abs(alpha) * k * 2 * eps;
        T err_beta = abs(beta) * eps;

        std::vector<T> A_abs(rows_A * cols_A);
        std::vector<T> B_abs(k * n);
        for (int64_t i = 0; i < (int64_t)A_abs.size(); ++i)
            A_abs[i] = abs(A_dense[i]);
        for (int64_t i = 0; i < (int64_t)B_abs.size(); ++i)
            B_abs[i] = abs(B_dense[i]);

        // Start error bound with |beta| * eps * |C_orig|
        std::vector<T> E(m * n, 0.0);
        if (beta != (T)0) {
            // C_orig was the random matrix generated above (before gemm overwrote C_ref).
            // We already copied it into C_actual before spgemm. For the error bound,
            // we need |C_orig|. Re-generate it.
            auto C_orig = std::get<0>(random_matrix<T>(m, n, RandBLAS::RNGState(42)));
            for (int64_t i = 0; i < m * n; ++i)
                E[i] = abs(C_orig[i]);
        }

        blas::gemm(layout, opA, Op::NoTrans, m, n, k,
            err_alpha, A_abs.data(), lda, B_abs.data(), ldb,
            err_beta, E.data(), ldc);

        test::comparison::buffs_approx_equal(
            C_actual.data(), C_ref.data(), E.data(), m * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    // Convenience: CSR x CSR
    void csr_times_csr(
        Op opA, int64_t m, int64_t n, int64_t k,
        T alpha, T beta, Layout layout,
        T p_A, T p_B, uint32_t key_A = 0, uint32_t key_B = 1
    ) {
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, opA);
        auto A = make_csr(rows_A, cols_A, p_A, key_A);
        auto B = make_csr(k, n, p_B, key_B);
        test_spgemm(opA, m, n, k, alpha, beta, A, B, layout);
    }

    // Convenience: COO x CSR
    void coo_times_csr(
        Op opA, int64_t m, int64_t n, int64_t k,
        T alpha, T beta, Layout layout,
        T p_A, T p_B, uint32_t key_A = 0, uint32_t key_B = 1
    ) {
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, opA);
        auto A = make_coo(rows_A, cols_A, p_A, key_A);
        auto B = make_csr(k, n, p_B, key_B);
        test_spgemm(opA, m, n, k, alpha, beta, A, B, layout);
    }

    // Convenience: CSR x COO
    void csr_times_coo(
        Op opA, int64_t m, int64_t n, int64_t k,
        T alpha, T beta, Layout layout,
        T p_A, T p_B, uint32_t key_A = 0, uint32_t key_B = 1
    ) {
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, opA);
        auto A = make_csr(rows_A, cols_A, p_A, key_A);
        auto B = make_coo(k, n, p_B, key_B);
        test_spgemm(opA, m, n, k, alpha, beta, A, B, layout);
    }
};

class TestSpGEMM_double : public TestSpGEMM<double> {};
class TestSpGEMM_float  : public TestSpGEMM<float> {};


////////////////////////////////////////////////////////////////////////
//
//      CSR x CSR: basic multiplication
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSpGEMM_double, csr_times_csr_tall_colmajor) {
    for (uint32_t key : {0}) {
        csr_times_csr(Op::NoTrans, 200, 30, 50, 1.0, 0.0, Layout::ColMajor, 0.01, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 200, 30, 50, 1.0, 0.0, Layout::ColMajor, 0.10, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 200, 30, 50, 1.0, 0.0, Layout::ColMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, csr_times_csr_tall_rowmajor) {
    for (uint32_t key : {0}) {
        csr_times_csr(Op::NoTrans, 200, 30, 50, 1.0, 0.0, Layout::RowMajor, 0.01, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 200, 30, 50, 1.0, 0.0, Layout::RowMajor, 0.10, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 200, 30, 50, 1.0, 0.0, Layout::RowMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, csr_times_csr_wide_colmajor) {
    for (uint32_t key : {0}) {
        csr_times_csr(Op::NoTrans, 30, 200, 50, 1.0, 0.0, Layout::ColMajor, 0.01, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 30, 200, 50, 1.0, 0.0, Layout::ColMajor, 0.10, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 30, 200, 50, 1.0, 0.0, Layout::ColMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, csr_times_csr_wide_rowmajor) {
    for (uint32_t key : {0}) {
        csr_times_csr(Op::NoTrans, 30, 200, 50, 1.0, 0.0, Layout::RowMajor, 0.01, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 30, 200, 50, 1.0, 0.0, Layout::RowMajor, 0.10, 0.10, key, key+1);
        csr_times_csr(Op::NoTrans, 30, 200, 50, 1.0, 0.0, Layout::RowMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, csr_times_csr_square_colmajor) {
    csr_times_csr(Op::NoTrans, 100, 100, 100, 1.0, 0.0, Layout::ColMajor, 0.05, 0.05);
}

////////////////////////////////////////////////////////////////////////
//
//      CSR x CSR: nontrivial alpha/beta
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSpGEMM_double, nontrivial_alpha_colmajor) {
    double alpha = 5.5;
    double beta = 0.0;
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.05, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.10, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.80, 0.80);
}

TEST_F(TestSpGEMM_double, nontrivial_alpha_rowmajor) {
    double alpha = 5.5;
    double beta = 0.0;
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.05, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.10, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.80, 0.80);
}

TEST_F(TestSpGEMM_double, nontrivial_alpha_beta_colmajor) {
    double alpha = 5.5;
    double beta = -1.0;
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.05, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.10, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.80, 0.80);
}

TEST_F(TestSpGEMM_double, nontrivial_alpha_beta_rowmajor) {
    double alpha = 5.5;
    double beta = -1.0;
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.05, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.10, 0.10);
    csr_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.80, 0.80);
}

////////////////////////////////////////////////////////////////////////
//
//      CSR x CSR: transpose of first operand
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSpGEMM_double, transpose_A_colmajor) {
    for (uint32_t key : {0}) {
        // A is k-by-m (before transpose), op(A) is m-by-k
        csr_times_csr(Op::Trans, 50, 30, 40, 1.0, 0.0, Layout::ColMajor, 0.01, 0.10, key, key+1);
        csr_times_csr(Op::Trans, 50, 30, 40, 1.0, 0.0, Layout::ColMajor, 0.10, 0.10, key, key+1);
        csr_times_csr(Op::Trans, 50, 30, 40, 1.0, 0.0, Layout::ColMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, transpose_A_rowmajor) {
    for (uint32_t key : {0}) {
        csr_times_csr(Op::Trans, 50, 30, 40, 1.0, 0.0, Layout::RowMajor, 0.01, 0.10, key, key+1);
        csr_times_csr(Op::Trans, 50, 30, 40, 1.0, 0.0, Layout::RowMajor, 0.10, 0.10, key, key+1);
        csr_times_csr(Op::Trans, 50, 30, 40, 1.0, 0.0, Layout::RowMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, transpose_A_alpha_beta_colmajor) {
    double alpha = 3.0;
    double beta = -2.0;
    csr_times_csr(Op::Trans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.10, 0.10);
    csr_times_csr(Op::Trans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.80, 0.80);
}

////////////////////////////////////////////////////////////////////////
//
//      Mixed formats: COO x CSR, CSR x COO
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSpGEMM_double, coo_times_csr_colmajor) {
    for (uint32_t key : {0}) {
        coo_times_csr(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::ColMajor, 0.10, 0.10, key, key+1);
        coo_times_csr(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::ColMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, coo_times_csr_rowmajor) {
    for (uint32_t key : {0}) {
        coo_times_csr(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::RowMajor, 0.10, 0.10, key, key+1);
        coo_times_csr(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::RowMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, csr_times_coo_colmajor) {
    for (uint32_t key : {0}) {
        csr_times_coo(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::ColMajor, 0.10, 0.10, key, key+1);
        csr_times_coo(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::ColMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, csr_times_coo_rowmajor) {
    for (uint32_t key : {0}) {
        csr_times_coo(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::RowMajor, 0.10, 0.10, key, key+1);
        csr_times_coo(Op::NoTrans, 100, 50, 60, 1.0, 0.0, Layout::RowMajor, 0.80, 0.80, key, key+1);
    }
}

TEST_F(TestSpGEMM_double, coo_times_csr_alpha_beta) {
    double alpha = 2.5;
    double beta = -0.5;
    coo_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::ColMajor, 0.10, 0.10);
    coo_times_csr(Op::NoTrans, 50, 30, 40, alpha, beta, Layout::RowMajor, 0.10, 0.10);
}

////////////////////////////////////////////////////////////////////////
//
//      Single precision
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSpGEMM_float, csr_times_csr_colmajor) {
    csr_times_csr(Op::NoTrans, 100, 50, 60, 1.0f, 0.0f, Layout::ColMajor, 0.10f, 0.10f);
    csr_times_csr(Op::NoTrans, 100, 50, 60, 1.0f, 0.0f, Layout::ColMajor, 0.80f, 0.80f);
}

TEST_F(TestSpGEMM_float, csr_times_csr_rowmajor) {
    csr_times_csr(Op::NoTrans, 100, 50, 60, 1.0f, 0.0f, Layout::RowMajor, 0.10f, 0.10f);
    csr_times_csr(Op::NoTrans, 100, 50, 60, 1.0f, 0.0f, Layout::RowMajor, 0.80f, 0.80f);
}

TEST_F(TestSpGEMM_float, csr_times_csr_alpha_beta) {
    csr_times_csr(Op::NoTrans, 50, 30, 40, 5.5f, -1.0f, Layout::ColMajor, 0.10f, 0.10f);
    csr_times_csr(Op::NoTrans, 50, 30, 40, 5.5f, -1.0f, Layout::RowMajor, 0.10f, 0.10f);
}

TEST_F(TestSpGEMM_float, csr_times_csr_transpose_A) {
    csr_times_csr(Op::Trans, 50, 30, 40, 1.0f, 0.0f, Layout::ColMajor, 0.10f, 0.10f);
    csr_times_csr(Op::Trans, 50, 30, 40, 1.0f, 0.0f, Layout::RowMajor, 0.10f, 0.10f);
}

#endif // RandBLAS_HAS_MKL
