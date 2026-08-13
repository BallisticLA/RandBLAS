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
#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>

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
        // Zero out the strict triangle NOT named by uplo (k=1 skips the
        // diagonal), leaving only the structurally-stored side + diagonal.
        auto other = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        RandBLAS::overwrite_triangle(Layout::ColMajor, other, n, 1, A, lda);
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
        uint32_t seed_A, uint32_t seed_B,
        bool route_via_wrapper = false
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

        // Under test: spsymm on the one-triangle sparse A. By default we
        // call the low-level dispatcher directly; if `route_via_wrapper` is
        // set, we go through the public RandBLAS::spsymm(Symmetric<SpMat>)
        // overload to exercise the wrapper-routing path. side=Left only
        // for the wrapper path since the public wrapper defaults side=Left.
        if (route_via_wrapper) {
            randblas_require(side == Side::Left);
            auto A_sym = RandBLAS::as_symmetric(A_sparse, uplo);
            RandBLAS::spsymm(layout, m_BY, n_BY,
                             alpha, A_sym, B.data(), ldb,
                             beta, Y_actual.data(), ldy);
        } else {
            RandBLAS::sparse_data::spsymm(layout, side, uplo, m_BY, n_BY,
                                          alpha, A_sparse, B.data(), ldb,
                                          beta, Y_actual.data(), ldy);
        }

        // Tolerance: the dense reference (blas::symm) and the sparse path
        // accumulate in different orders, so we get a few ULPs of accumulation
        // divergence. Use 100*eps absolute tolerance to absorb that.
        T atol = T(100) * std::numeric_limits<T>::epsilon();
        T rtol = T(10) * std::numeric_limits<T>::epsilon();
        auto msg = RandBLAS::testing::matrices_approx_equal(
            layout, blas::Op::NoTrans, m_BY, n_BY,
            Y_actual.data(), ldy, Y_expect.data(), ldy,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
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

    // Case D: sparse-symmetric A times sparse B -> dense Y. Reference is
    // dense blas::symm on a fully-populated A and a densified B.
    template <typename SpMatA, typename SpMatB, typename T = typename SpMatA::scalar_t>
    static void run_case_d(
        Layout layout, Side side, Uplo uplo,
        int64_t n_A, int64_t d,
        T alpha, T beta,
        uint32_t seed_A, uint32_t seed_B,
        double density_B = 0.3
    ) {
        int64_t m_BY, n_BY;
        if (side == Side::Left) { m_BY = n_A; n_BY = d; }
        else                    { m_BY = d;   n_BY = n_A; }

        // Build dense symm A (full-storage reference).
        int64_t lda = n_A;
        std::vector<T> A_full(lda * n_A, T(0));
        fill_sym_dense<T>(n_A, A_full.data(), lda, seed_A);
        // Build sparse A from a one-triangle copy.
        std::vector<T> A_tri(A_full);
        zero_other_triangle<T>(n_A, A_tri.data(), lda, uplo);
        SpMatA A_sparse(n_A, n_A);
        dense_to_sparse_format<SpMatA, T>(Layout::ColMajor, A_tri.data(), T(0), A_sparse);

        // Random sparse B as a ColMajor dense buffer first, then convert to SpMatB.
        std::vector<T> B_dense(m_BY * n_BY, T(0));
        {
            std::mt19937_64 rng(static_cast<uint64_t>(seed_B));
            std::uniform_real_distribution<double> uni01(0.0, 1.0);
            std::uniform_real_distribution<double> univ(-1.0, 1.0);
            for (int64_t j = 0; j < n_BY; ++j) {
                for (int64_t i = 0; i < m_BY; ++i) {
                    if (uni01(rng) < density_B) {
                        B_dense[i + j * m_BY] = static_cast<T>(univ(rng));
                    }
                }
            }
        }
        SpMatB B_sparse(m_BY, n_BY);
        dense_to_sparse_format<SpMatB, T>(Layout::ColMajor, B_dense.data(), T(0), B_sparse);

        int64_t ldb = (layout == Layout::ColMajor) ? m_BY : n_BY;
        int64_t ldy = ldb;
        // The dense reference call to blas::symm uses the requested layout.
        std::vector<T> B_dense_layout(m_BY * n_BY);
        if (layout == Layout::ColMajor) {
            std::copy(B_dense.begin(), B_dense.end(), B_dense_layout.begin());
        } else {
            for (int64_t i = 0; i < m_BY; ++i)
                for (int64_t j = 0; j < n_BY; ++j)
                    B_dense_layout[i * ldb + j] = B_dense[i + j * m_BY];
        }

        std::vector<T> Y_actual(m_BY * n_BY);
        DenseDist DY(m_BY, n_BY, ScalarDist::Uniform);
        RandBLAS::fill_dense_unpacked(layout, DY, m_BY, n_BY, 0, 0, Y_actual.data(), RNGState(seed_B + 13));
        std::vector<T> Y_expect = Y_actual;

        // Reference: dense blas::symm on full-storage A and dense B.
        blas::symm(layout, side, uplo, m_BY, n_BY,
                   alpha, A_full.data(), lda, B_dense_layout.data(), ldb,
                   beta, Y_expect.data(), ldy);

        // Under test: sparse-symm A times sparse B via Case D.
        RandBLAS::sparse_data::spsymm(layout, side, uplo, m_BY, n_BY,
                                      alpha, A_sparse, B_sparse,
                                      beta, Y_actual.data(), ldy);

        T atol = T(100) * std::numeric_limits<T>::epsilon();
        T rtol = T(10)  * std::numeric_limits<T>::epsilon();
        auto msg = RandBLAS::testing::matrices_approx_equal(
            layout, blas::Op::NoTrans, m_BY, n_BY,
            Y_actual.data(), ldy, Y_expect.data(), ldy,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        if (!msg.empty()) FAIL() << msg;
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


// Format-pair sweep: 3 A-formats x 3 B-formats x both sides + an uplo and
// edge-case sample. MKL handles all 9 pairs via make_mkl_handle.
TEST_F(TestSpsymm, CaseD_CSR_CSR_Left) {
    run_case_d<CSRMatrix<double>, CSRMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 1.5, -0.5, 0, 1);
    run_case_d<CSRMatrix<double>, CSRMatrix<double>>(Layout::RowMajor, Side::Left, Uplo::Lower, 8, 3, 1.5, -0.5, 0, 1);
}
TEST_F(TestSpsymm, CaseD_CSC_CSC_Left) {
    run_case_d<CSCMatrix<double>, CSCMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 1.5, -0.5, 0, 1);
    run_case_d<CSCMatrix<double>, CSCMatrix<double>>(Layout::RowMajor, Side::Left, Uplo::Lower, 8, 3, 1.5, -0.5, 0, 1);
}
TEST_F(TestSpsymm, CaseD_COO_COO_Left) {
    run_case_d<COOMatrix<double>, COOMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 1.5, -0.5, 0, 1);
    run_case_d<COOMatrix<double>, COOMatrix<double>>(Layout::RowMajor, Side::Left, Uplo::Lower, 8, 3, 1.5, -0.5, 0, 1);
}
TEST_F(TestSpsymm, CaseD_Mixed_Format_Left) {
    run_case_d<CSRMatrix<double>, CSCMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 1.0, 0.0, 0, 1);
    run_case_d<CSCMatrix<double>, CSRMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Lower, 8, 3, 1.0, 0.0, 0, 1);
    run_case_d<COOMatrix<double>, CSRMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 1.0, 0.0, 0, 1);
}
TEST_F(TestSpsymm, CaseD_CSR_CSR_Right) {
    run_case_d<CSRMatrix<double>, CSRMatrix<double>>(Layout::ColMajor, Side::Right, Uplo::Upper, 8, 3, 1.5, -0.5, 0, 1);
    run_case_d<CSRMatrix<double>, CSRMatrix<double>>(Layout::RowMajor, Side::Right, Uplo::Lower, 8, 3, 1.5, -0.5, 0, 1);
}
TEST_F(TestSpsymm, CaseD_Float) {
    run_case_d<CSRMatrix<float>, CSRMatrix<float>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 1.5f, -0.5f, 0, 1);
}
TEST_F(TestSpsymm, CaseD_AlphaZero_BetaScale) {
    // alpha=0 path: just beta-scales Y, doesn't even touch A or B.
    run_case_d<CSRMatrix<double>, CSRMatrix<double>>(Layout::ColMajor, Side::Left, Uplo::Upper, 8, 3, 0.0, 0.5, 0, 1);
}

// Routes through the public RandBLAS::spsymm(Symmetric<SpMat>) wrapper
// overload instead of the lower-level RandBLAS::sparse_data::spsymm.
// All other setup (dense reference, comparison tolerance) is identical
// to run_case; we set route_via_wrapper=true to flip the dispatch.
TEST_F(TestSpsymm, SymmetricWrapper) {
    run_case<CSRMatrix<double>>(
        Layout::ColMajor, Side::Left, Uplo::Upper,
        /*n_A=*/8, /*d=*/3,
        /*alpha=*/1.0, /*beta=*/0.0,
        /*seed_A=*/0, /*seed_B=*/11,
        /*route_via_wrapper=*/true
    );
}


// MARK: ERROR PATHS

TEST_F(TestSpsymm, leading_dim_too_small_throws) {
    int64_t n_A = 6, d = 3;
    std::vector<double> A_tri(n_A * n_A, 0.0);
    fill_sym_dense<double>(n_A, A_tri.data(), n_A, 99);
    zero_other_triangle<double>(n_A, A_tri.data(), n_A, Uplo::Upper);
    CSRMatrix<double> A_sparse(n_A, n_A);
    dense_to_sparse_format<CSRMatrix<double>, double>(Layout::ColMajor, A_tri.data(), 0.0, A_sparse);

    std::vector<double> B(n_A * d, 1.0), Y(n_A * d, 0.0);
    // side=Left, ColMajor: B and Y are n_A-by-d, so ldb, ldy >= n_A.
    ASSERT_THROW(
        RandBLAS::sparse_data::spsymm(Layout::ColMajor, Side::Left, Uplo::Upper, n_A, d,
                                      1.0, A_sparse, B.data(), n_A, 0.0, Y.data(), n_A - 1),
        RandBLAS::Error);
    ASSERT_THROW(
        RandBLAS::sparse_data::spsymm(Layout::ColMajor, Side::Left, Uplo::Upper, n_A, d,
                                      1.0, A_sparse, B.data(), n_A - 1, 0.0, Y.data(), n_A),
        RandBLAS::Error);
}

TEST_F(TestSpsymm, one_based_indices_throw) {
    // The COOMatrix expert constructor accepts IndexBase::One, but spsymm
    // requires zero-based indices (as left_spmm does).
    double  vals[] = {2.0, 1.0, 3.0};
    int64_t rows[] = {1, 1, 2};
    int64_t cols[] = {1, 2, 2};
    COOMatrix<double> A(2, 2, 3, vals, rows, cols, true,
                        RandBLAS::sparse_data::IndexBase::One);
    std::vector<double> B(2 * 2, 1.0), Y(2 * 2, 0.0);
    ASSERT_THROW(
        RandBLAS::sparse_data::spsymm(Layout::ColMajor, Side::Left, Uplo::Upper, 2, 2,
                                      1.0, A, B.data(), 2, 0.0, Y.data(), 2),
        RandBLAS::Error);
}

// Case D with int32 indices: exercises whichever branch the build selects
// (expand-A + spgemm when the index width matches MKL_INT, the densify-B
// composition otherwise). Both must produce the same answer.
TEST_F(TestSpsymm, CaseD_Int32_Indices) {
    // A_sym = [[2, 1, 0], [1, 3, 0], [0, 0, 4]], upper triangle stored.
    double  a_vals[] = {2.0, 1.0, 3.0, 4.0};
    int32_t a_rows[] = {0, 0, 1, 2};
    int32_t a_cols[] = {0, 1, 1, 2};
    COOMatrix<double, int32_t> A(3, 3, 4, a_vals, a_rows, a_cols);
    // B = [[1, 0], [0, 5], [2, 0]].
    double  b_vals[] = {1.0, 5.0, 2.0};
    int32_t b_rows[] = {0, 1, 2};
    int32_t b_cols[] = {0, 1, 0};
    COOMatrix<double, int32_t> B(3, 2, 3, b_vals, b_rows, b_cols);

    // A_sym * B = [[2, 5], [1, 15], [8, 0]].
    std::vector<double> Y(3 * 2, 0.0);
    RandBLAS::sparse_data::spsymm(Layout::ColMajor, Side::Left, Uplo::Upper, 3, 2,
                                  1.0, A, B, 0.0, Y.data(), 3);
    std::vector<double> expect = {2.0, 1.0, 8.0, 5.0, 15.0, 0.0};
    for (size_t i = 0; i < expect.size(); ++i)
        EXPECT_NEAR(Y[i], expect[i], 1e-14) << "mismatch at flat index " << i;
}
