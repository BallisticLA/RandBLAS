// Copyright, 2026. See LICENSE for copyright holder information.
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

#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/spsymm_dispatch.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/testing/comparison.hh"

#include <gtest/gtest.h>
#include <vector>
#include <type_traits>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::coo;
using namespace RandBLAS::sparse_data::csr;
using namespace RandBLAS::sparse_data::csc;
using blas::Layout;
using blas::Side;
using blas::Uplo;
using RandBLAS::Axis;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using RandBLAS::ScalarDist;


class TestSpsymm : public ::testing::Test {
protected:
    template <typename T>
    static void fill_sym_dense(int64_t n, T* A, int64_t lda, uint32_t seed) {
        DenseDist D(lda, lda, ScalarDist::Uniform);
        RandBLAS::fill_dense_unpacked(Layout::ColMajor, D, n, n, 0, 0, A, RNGState(seed));
        RandBLAS::symmetrize(Layout::ColMajor, Uplo::Upper, n, A, lda);
    }

    template <typename T>
    static void zero_other_triangle(int64_t n, T* A, int64_t lda, Uplo uplo) {
        // Zero out the strict triangle NOT named by uplo, leaving only the
        // structurally-stored side + diagonal.
        if (uplo == Uplo::Upper) {
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = j + 1; i < n; ++i)
                    A[i + j * lda] = T(0);
        } else {
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < j; ++i)
                    A[i + j * lda] = T(0);
        }
    }

    template <typename SpMat, typename T>
    static void dense_to_sparse_format(Layout layout, T* dense_buf, T abs_tol, SpMat& sp) {
        using sint_t = typename SpMat::index_t;
        if constexpr (std::is_same_v<SpMat, COOMatrix<T, sint_t>>) {
            dense_to_coo(layout, dense_buf, abs_tol, sp);
        } else if constexpr (std::is_same_v<SpMat, CSRMatrix<T, sint_t>>) {
            dense_to_csr(layout, dense_buf, abs_tol, sp);
        } else if constexpr (std::is_same_v<SpMat, CSCMatrix<T, sint_t>>) {
            dense_to_csc(layout, dense_buf, abs_tol, sp);
        } else {
            static_assert(sizeof(SpMat) == 0, "Unsupported sparse format.");
        }
    }

    template <typename SpMat, typename T = typename SpMat::scalar_t>
    static void run_case(
        Layout layout, Side side, Uplo uplo,
        int64_t n_A, int64_t d,
        T alpha, T beta,
        uint32_t seed_A, uint32_t seed_B
    ) {
        // For side=Left:  Y = alpha*A*B + beta*Y, A is n_A x n_A, B and Y are n_A x d.
        // For side=Right: Y = alpha*B*A + beta*Y, B and Y are d x n_A, A is n_A x n_A.
        int64_t m_BY, n_BY;
        if (side == Side::Left) { m_BY = n_A; n_BY = d; }
        else                    { m_BY = d;   n_BY = n_A; }

        // Build dense symmetric A.
        int64_t lda = n_A;
        std::vector<T> A_full(lda * n_A, T(0));
        fill_sym_dense<T>(n_A, A_full.data(), lda, seed_A);

        // Build sparse A: take A_full, zero out the non-named triangle, then convert.
        std::vector<T> A_triangle(A_full);
        zero_other_triangle<T>(n_A, A_triangle.data(), lda, uplo);
        SpMat A_sparse(n_A, n_A);
        dense_to_sparse_format<SpMat, T>(Layout::ColMajor, A_triangle.data(), T(0), A_sparse);

        // Build random B and an initial Y (for beta != 0 to be non-trivial).
        int64_t ldb = (layout == Layout::ColMajor) ? m_BY : n_BY;
        int64_t ldy = ldb;
        std::vector<T> B(m_BY * n_BY);
        DenseDist DB(m_BY, n_BY, ScalarDist::Uniform);
        RandBLAS::fill_dense_unpacked(layout, DB, m_BY, n_BY, 0, 0, B.data(), RNGState(seed_B));

        std::vector<T> Y_actual(m_BY * n_BY);
        RandBLAS::fill_dense_unpacked(layout, DB, m_BY, n_BY, 0, 0, Y_actual.data(), RNGState(seed_B + 7));
        std::vector<T> Y_expect = Y_actual;

        // Reference using dense blas::symm on the fully-populated A_full
        // (both triangles match because A_full is symmetrized; choice of uplo
        // for the reference doesn't matter, but we pass `uplo` for consistency).
        blas::symm(layout, side, uplo, m_BY, n_BY,
                   alpha, A_full.data(), lda, B.data(), ldb,
                   beta, Y_expect.data(), ldy);

        // Under test: spsymm on the one-triangle sparse A.
        RandBLAS::sparse_data::spsymm(layout, side, uplo, m_BY, n_BY,
                                      alpha, A_sparse, B.data(), ldb,
                                      beta, Y_actual.data(), ldy);

        // Tolerance: the dense reference (blas::symm) and the sparse path
        // accumulate in different orders, so we get a few ULPs of accumulation
        // divergence. Use 100*eps absolute tolerance to absorb that.
        T atol = T(100) * std::numeric_limits<T>::epsilon();
        T rtol = T(10) * std::numeric_limits<T>::epsilon();
        auto msg = RandBLAS::testing::matrices_approx_equal(
            layout, blas::Op::NoTrans, m_BY, n_BY,
            Y_actual.data(), ldy, Y_expect.data(), ldy,
            __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        if (!msg.empty()) FAIL() << msg;
    }

    template <typename SpMat>
    static void sweep_layout_uplo(Side side, int64_t n_A, int64_t d, double alpha, double beta) {
        using T = typename SpMat::scalar_t;
        for (auto layout : {Layout::ColMajor, Layout::RowMajor}) {
            for (auto uplo : {Uplo::Upper, Uplo::Lower}) {
                SCOPED_TRACE(testing::Message() <<
                    "layout=" << (layout == Layout::ColMajor ? "Col" : "Row")
                    << " uplo=" << (uplo == Uplo::Upper ? "U" : "L"));
                run_case<SpMat>(layout, side, uplo, n_A, d, T(alpha), T(beta), 0, 1);
            }
        }
    }
};


// 24-cell coverage: {COO, CSR, CSC} x {ColMajor, RowMajor} x {Upper, Lower} x {Left, Right}.
// Each TEST_F sweeps the (layout, uplo) plane internally for one (format, side) combination.

TEST_F(TestSpsymm, CSR_Left)  { sweep_layout_uplo<CSRMatrix<double>>(Side::Left,  10, 4, 1.5, -0.5); }
TEST_F(TestSpsymm, CSR_Right) { sweep_layout_uplo<CSRMatrix<double>>(Side::Right, 10, 4, 1.5, -0.5); }
TEST_F(TestSpsymm, CSC_Left)  { sweep_layout_uplo<CSCMatrix<double>>(Side::Left,  10, 4, 1.5, -0.5); }
TEST_F(TestSpsymm, CSC_Right) { sweep_layout_uplo<CSCMatrix<double>>(Side::Right, 10, 4, 1.5, -0.5); }
TEST_F(TestSpsymm, COO_Left)  { sweep_layout_uplo<COOMatrix<double>>(Side::Left,  10, 4, 1.5, -0.5); }
TEST_F(TestSpsymm, COO_Right) { sweep_layout_uplo<COOMatrix<double>>(Side::Right, 10, 4, 1.5, -0.5); }

// Float coverage for one representative format
TEST_F(TestSpsymm, CSR_Left_Float)  { sweep_layout_uplo<CSRMatrix<float>>(Side::Left,  10, 4, 1.5, -0.5); }

// beta=0 (init-from-zero) edge case
TEST_F(TestSpsymm, CSR_Left_Beta0) {
    for (auto layout : {Layout::ColMajor, Layout::RowMajor})
        for (auto uplo : {Uplo::Upper, Uplo::Lower})
            run_case<CSRMatrix<double>>(layout, Side::Left, uplo, 10, 4, 1.0, 0.0, 0, 2);
}

// alpha=0 (only beta-scaling) edge case
TEST_F(TestSpsymm, CSR_Left_Alpha0) {
    run_case<CSRMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 10, 4, 0.0, 0.5, 0, 3);
}

// Symmetric<SpMat> wrapper routing: covers the public RandBLAS::spsymm(layout, Symmetric, ...) overload
TEST_F(TestSpsymm, SymmetricWrapper) {
    using SpMat = CSRMatrix<double>;
    int64_t n_A = 8;
    int64_t d = 3;
    int64_t lda = n_A;
    std::vector<double> A_full(lda * n_A, 0.0);
    fill_sym_dense<double>(n_A, A_full.data(), lda, 0);
    std::vector<double> A_tri(A_full);
    zero_other_triangle<double>(n_A, A_tri.data(), lda, Uplo::Upper);
    SpMat A_sparse(n_A, n_A);
    dense_to_csr(Layout::ColMajor, A_tri.data(), 0.0, A_sparse);

    int64_t m_BY = n_A, n_BY = d;
    int64_t ldb = m_BY, ldy = m_BY;
    std::vector<double> B(m_BY * n_BY);
    DenseDist DB(m_BY, n_BY, ScalarDist::Uniform);
    RandBLAS::fill_dense_unpacked(Layout::ColMajor, DB, m_BY, n_BY, 0, 0, B.data(), RNGState(11));
    std::vector<double> Y_actual(m_BY * n_BY, 0.0);
    std::vector<double> Y_expect = Y_actual;

    double alpha = 1.0, beta = 0.0;
    blas::symm(Layout::ColMajor, Side::Left, Uplo::Upper, m_BY, n_BY,
               alpha, A_full.data(), lda, B.data(), ldb,
               beta, Y_expect.data(), ldy);

    // Route via wrapper overload at the public RandBLAS:: namespace
    auto A_sym = RandBLAS::as_symmetric(A_sparse, Uplo::Upper);
    RandBLAS::spsymm(Layout::ColMajor, m_BY, n_BY, alpha, A_sym,
                     B.data(), ldb, beta, Y_actual.data(), ldy);

    double atol = 100.0 * std::numeric_limits<double>::epsilon();
    double rtol = 10.0 * std::numeric_limits<double>::epsilon();
    auto msg = RandBLAS::testing::matrices_approx_equal(
        Layout::ColMajor, blas::Op::NoTrans, m_BY, n_BY,
        Y_actual.data(), ldy, Y_expect.data(), ldy,
        __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
    );
    if (!msg.empty()) FAIL() << msg;
}
