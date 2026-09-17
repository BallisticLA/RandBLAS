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
#include "RandBLAS/base.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/sksy.hh"

using blas::Layout;
using blas::Uplo;
using RandBLAS::ScalarDist;
using RandBLAS::DenseDist;
using RandBLAS::DenseSkOp;
using RandBLAS::SparseDist;
using RandBLAS::SparseSkOp;
using RandBLAS::RNGState;
using RandBLAS::Axis;

#include "RandBLAS/testing/comparison.hh"

#include <gtest/gtest.h>
#include <vector>


template <typename T, typename STATE>
void random_symmetric_mat(int64_t n, T* A, int64_t lda, STATE s) {
    // This function can be interpreted as first generating a random lda-by-lda symmetric matrix
    // whose entries in the upper triangle are iid, then symmetrizing that matrix, then
    // zeroing out all entries outside the leading principal submatrix of order n.
    RandBLAS::fill_dense_unpacked(Layout::ColMajor, {lda, lda}, n, n, 0, 0, A, s);
    RandBLAS::symmetrize(Layout::ColMajor, Uplo::Upper, n, A, lda);
    return;
}

template <typename T, typename SKOP>
blas::Side sketch_symmetric_side(
    blas::Side side_skop, blas::Layout layout, blas::Uplo uplo,
    int64_t rows_out, int64_t cols_out,
    T alpha, const T* A, int64_t lda, SKOP &S, int64_t ro_s, int64_t co_s, T beta, T* B, int64_t ldb
) {
    if (side_skop == blas::Side::Left) {
        RandBLAS::sketch_symmetric(layout, uplo, rows_out, cols_out, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
        return blas::Side::Right;
    } else {
        RandBLAS::sketch_symmetric(layout, uplo, rows_out, cols_out, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
        return blas::Side::Left;
    }
}

RandBLAS::dims64_t dims_of_sketch_symmetric_output(int64_t d, int64_t n, blas::Side side_skop) {
    // n    : dimensional parameter for the n-x-n symmetric matrices used in tests below.
    // d    : the embedding dimension for the sketching operator (d < n for sketching and d > n for lifting).
    // side : Left if the sketching operator multiplies the n-x-n matrix from the left; Right otherwise. 
    if (side_skop == blas::Side::Left) {
        return {d, n};
    } else {
        return {n, d};
    }
}

class TestSketchSymmetric : public ::testing::Test {
    protected: 

    template <typename T>
    static void test_same_layouts(
        uint32_t seed_a, uint32_t seed_skop, Axis major_axis, T alpha, int64_t d, int64_t n, int64_t lda, T beta, blas::Side side_skop,
        blas::Uplo uplo = blas::Uplo::Upper
    ) {
        auto [rows_out, cols_out] = dims_of_sketch_symmetric_output(d, n, side_skop);
        std::vector<T> A(lda*lda, 0.0);
        random_symmetric_mat(n, A.data(), lda, RNGState(seed_a));
        DenseDist D(rows_out, cols_out, ScalarDist::Uniform, major_axis);
        DenseSkOp<T> S(D, seed_skop);
        RandBLAS::fill_dense(S);
        int64_t lds = (S.layout == Layout::RowMajor) ? cols_out : rows_out;
        int64_t ldb = lds;
        uint32_t seed_b = seed_a + 42;
        std::vector<T> B_actual(d*n);
        RandBLAS::fill_dense(D, B_actual.data(), RNGState(seed_b));
        std::vector<T> B_expect(B_actual);

        // Compute the actual output
        auto side_a = sketch_symmetric_side(side_skop, S.layout, uplo, rows_out, cols_out, alpha, A.data(), lda, S, 0, 0, beta, B_actual.data(), ldb);
        // Compute the expected output
        blas::symm(S.layout, side_a, uplo, rows_out, cols_out, alpha, A.data(), lda, S.buff, lds, beta, B_expect.data(), ldb);

        auto msg = RandBLAS::testing::matrices_approx_equal(
            S.layout, blas::Op::NoTrans, rows_out, cols_out, B_actual.data(), ldb, B_expect.data(), ldb,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    template <typename T>
    static void test_opposing_layouts(
        uint32_t seed_a, uint32_t seed_skop, Axis major_axis, T alpha, int64_t d, int64_t n, int64_t lda, T beta, blas::Side side_skop,
        blas::Uplo uplo = blas::Uplo::Upper
    ) {
        auto [rows_out, cols_out] = dims_of_sketch_symmetric_output(d, n, side_skop);
        std::vector<T> A(lda*lda, 0.0);
        random_symmetric_mat(n, A.data(), lda, RNGState(seed_a));
        DenseDist D(rows_out, cols_out, ScalarDist::Uniform, major_axis);
        DenseSkOp<T> S(D, seed_skop);
        RandBLAS::fill_dense(S);
        int64_t lds_init, ldb;
        Layout layout_B;
        if (S.layout == Layout::RowMajor) {
            layout_B = Layout::ColMajor;
            ldb = rows_out;
            lds_init = cols_out;
        } else {
            layout_B = Layout::RowMajor;
            ldb = cols_out;
            lds_init = rows_out;
        }
        uint32_t seed_b = seed_a + 42;
        std::vector<T> B_actual(d*n);
        RandBLAS::fill_dense(D, B_actual.data(), RNGState(seed_b));
        std::vector<T> B_expect(B_actual);
        // Compute the actual output
        auto side_a = sketch_symmetric_side(side_skop, layout_B, uplo, rows_out, cols_out, alpha, A.data(), lda, S, 0, 0, beta, B_actual.data(), ldb);
        // Compute the expected output
        std::vector<T> S_flipped(S.buff, S.buff + d*n);
        RandBLAS::util::flip_layout(S.layout, rows_out, cols_out, S_flipped, lds_init, ldb);
        blas::symm(layout_B, side_a, uplo, rows_out, cols_out, alpha, A.data(), lda, S_flipped.data(), ldb, beta, B_expect.data(), ldb);

        auto msg = RandBLAS::testing::matrices_approx_equal(
            layout_B, blas::Op::NoTrans, rows_out, cols_out, B_actual.data(), ldb, B_expect.data(), ldb,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    // Note on symmetry checking: the blas::Uplo overloads read only the
    // named triangle, so they perform no runtime symmetry check; the legacy
    // sym_check_tol overloads retain the check (covered under MARK: LEGACY
    // OVERLOADS below). Error paths for the Uplo overloads (leading dims,
    // submatrix window bounds) are covered under MARK: ERROR PATHS.

    // =============================================================================
    // Case B exerciser: sparse SkOp x dense symmetric A. Reference is the
    // densified-sparse-skop fed to blas::symm. layout is explicit (SparseSkOp
    // has no S.layout the way DenseSkOp does; the COO storage is layout-
    // agnostic, and the test layout determines how both B and the dense
    // reference of S are laid out). With materialize=false, S is handed to
    // sketch_symmetric unmaterialized (nnz < 0), exercising the
    // submatrix_as_coo window-sampling path; the reference is always built
    // from a materialized twin (same dist and seed give identical samples).
    template <typename T>
    static void test_sparse_skop(
        Layout layout,
        uint32_t seed_a, uint32_t seed_skop, Axis major_axis,
        T alpha, int64_t d, int64_t n, int64_t lda, T beta,
        blas::Side side_skop,
        int64_t vec_nnz = 2,
        Uplo uplo = Uplo::Upper,
        bool materialize = true
    ) {
        auto [rows_out, cols_out] = dims_of_sketch_symmetric_output(d, n, side_skop);
        std::vector<T> A(lda * lda, T(0));
        random_symmetric_mat(n, A.data(), lda, RNGState(seed_a));

        SparseDist DS(rows_out, cols_out, vec_nnz, major_axis);
        SparseSkOp<T> S(DS, seed_skop);
        if (materialize)
            RandBLAS::fill_sparse(S);
        SparseSkOp<T> S_ref(DS, seed_skop);
        RandBLAS::fill_sparse(S_ref);

        // Densify the reference twin into a buffer matching the requested
        // layout, for the SYMM reference. lds is the major-axis leading dim
        // (tight, so coo_to_dense's layout overload applies directly).
        int64_t lds = (layout == Layout::ColMajor) ? rows_out : cols_out;
        int64_t ldb = lds;
        std::vector<T> S_dense(static_cast<size_t>(rows_out) * cols_out, T(0));
        auto Scoo = RandBLAS::coo_view_of_skop(S_ref);
        RandBLAS::sparse_data::coo::coo_to_dense(Scoo, layout, S_dense.data());

        uint32_t seed_b = seed_a + 42;
        std::vector<T> B_actual(static_cast<size_t>(rows_out) * cols_out);
        DenseDist DB(rows_out, cols_out, ScalarDist::Uniform);
        RandBLAS::fill_dense_unpacked(layout, DB, rows_out, cols_out, 0, 0, B_actual.data(), RNGState(seed_b));
        std::vector<T> B_expect = B_actual;

        auto side_a = sketch_symmetric_side(
            side_skop, layout, uplo, rows_out, cols_out,
            alpha, A.data(), lda, S, 0, 0, beta, B_actual.data(), ldb
        );
        blas::symm(layout, side_a, uplo, rows_out, cols_out,
                   alpha, A.data(), lda, S_dense.data(), lds, beta, B_expect.data(), ldb);

        // Same FMA-order-divergence tolerance as the spsymm tests: the
        // column-driven sparse kernel accumulates in a different order than
        // dense SYMM on a fully-stored matrix.
        T atol = T(100) * std::numeric_limits<T>::epsilon();
        T rtol = T(10)  * std::numeric_limits<T>::epsilon();
        auto msg = RandBLAS::testing::matrices_approx_equal(
            layout, blas::Op::NoTrans, rows_out, cols_out,
            B_actual.data(), ldb, B_expect.data(), ldb,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol
        );
        if (!msg.empty()) FAIL() << msg;
    }
};


// MARK: SAME LAYOUTS

TEST_F(TestSketchSymmetric, left_sketch_10_to_3_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, left_lift_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, right_sketch_10_to_3_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
}

TEST_F(TestSketchSymmetric, right_lift_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
}


// MARK: OPPOSING LAYOUTS

TEST_F(TestSketchSymmetric, left_sketch_10_to_3_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
    // Finally, dispatch float template instantiations for full(er) test coverage.
    //
    //    The codepaths we're trying to hit don't differ if right-sketching,
    //    or if "lifting" rather than sketching, so the lines below are unique to
    //    this fixture.
    //
    test_opposing_layouts( 0,  1, Axis::Short, (float)0.5, 3, 10, 10, (float)0.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, Axis::Long,  (float)0.5, 3, 10, 10, (float)0.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, left_lift_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, right_sketch_10_to_3_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
}


TEST_F(TestSketchSymmetric, right_lift_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, Axis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
}


// MARK: SPARSE SkOp (Case B). 4-axis sweep:
//   side x layout x uplo x beta-mode (zero/nonzero).
// One seed pair per cell to keep the per-config trial count modest.

TEST_F(TestSketchSymmetric, sparse_skop_left_colmajor_upper) {
    test_sparse_skop<double>(Layout::ColMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10,  0.0, blas::Side::Left,  2, Uplo::Upper);
    test_sparse_skop<double>(Layout::ColMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left,  2, Uplo::Upper);
    test_sparse_skop<double>(Layout::ColMajor, 31, 33, Axis::Long,  0.5, 3, 10, 19,  0.0, blas::Side::Left,  2, Uplo::Upper);
    test_sparse_skop<float>( Layout::ColMajor,  0,  1, Axis::Short, 0.5f, 3, 10, 10, 0.0f, blas::Side::Left, 2, Uplo::Upper);
}

TEST_F(TestSketchSymmetric, sparse_skop_left_rowmajor_upper) {
    test_sparse_skop<double>(Layout::RowMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10,  0.0, blas::Side::Left,  2, Uplo::Upper);
    test_sparse_skop<double>(Layout::RowMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left,  2, Uplo::Upper);
    test_sparse_skop<double>(Layout::RowMajor, 31, 33, Axis::Long,  0.5, 3, 10, 19,  0.0, blas::Side::Left,  2, Uplo::Upper);
}

TEST_F(TestSketchSymmetric, sparse_skop_right_colmajor_upper) {
    test_sparse_skop<double>(Layout::ColMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10,  0.0, blas::Side::Right, 2, Uplo::Upper);
    test_sparse_skop<double>(Layout::ColMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right, 2, Uplo::Upper);
    test_sparse_skop<double>(Layout::ColMajor, 31, 33, Axis::Long,  0.5, 3, 10, 19,  0.0, blas::Side::Right, 2, Uplo::Upper);
}

TEST_F(TestSketchSymmetric, sparse_skop_right_rowmajor_upper) {
    test_sparse_skop<double>(Layout::RowMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10,  0.0, blas::Side::Right, 2, Uplo::Upper);
    test_sparse_skop<double>(Layout::RowMajor,  0,  1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right, 2, Uplo::Upper);
    test_sparse_skop<double>(Layout::RowMajor, 31, 33, Axis::Long,  0.5, 3, 10, 19,  0.0, blas::Side::Right, 2, Uplo::Upper);
}

TEST_F(TestSketchSymmetric, sparse_skop_lower_triangle) {
    // Uplo::Lower coverage: one cell per (side, layout) combo, since the
    // Upper-vs-Lower difference exercises the same kernel branches as
    // ColMajor-vs-RowMajor (different symmetric-read resolution).
    test_sparse_skop<double>(Layout::ColMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left,  2, Uplo::Lower);
    test_sparse_skop<double>(Layout::RowMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left,  2, Uplo::Lower);
    test_sparse_skop<double>(Layout::ColMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right, 2, Uplo::Lower);
    test_sparse_skop<double>(Layout::RowMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right, 2, Uplo::Lower);
}

TEST_F(TestSketchSymmetric, sparse_skop_lift) {
    // Embedding goes the wrong way (d > n): tests that the kernel handles
    // non-square sketches in both directions.
    test_sparse_skop<double>(Layout::ColMajor, 0, 1, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left,  2, Uplo::Upper);
    test_sparse_skop<double>(Layout::ColMajor, 0, 1, Axis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right, 2, Uplo::Upper);
}


TEST_F(TestSketchSymmetric, sparse_skop_unmaterialized) {
    // S is handed to sketch_symmetric with nnz < 0: the wrapper samples only
    // the requested window via submatrix_as_coo instead of materializing S.
    test_sparse_skop<double>(Layout::ColMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left,  2, Uplo::Upper, /*materialize=*/false);
    test_sparse_skop<double>(Layout::RowMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left,  2, Uplo::Lower, /*materialize=*/false);
    test_sparse_skop<double>(Layout::ColMajor, 0, 1, Axis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right, 2, Uplo::Upper, /*materialize=*/false);
}


// MARK: ERROR PATHS

TEST_F(TestSketchSymmetric, sparse_skop_bad_arguments_throw) {
    int64_t d = 3, n = 10;
    SparseDist DS(d, n, 2, Axis::Short);
    SparseSkOp<double> S(DS, 11);
    RandBLAS::fill_sparse(S);
    std::vector<double> A(n * n, 1.0);
    std::vector<double> B(static_cast<size_t>(d) * n, 0.0);

    // ldb below its ColMajor lower bound (B is d-by-n, so ldb >= d).
    ASSERT_THROW(
        RandBLAS::sketch_symmetric(Layout::ColMajor, Uplo::Upper, d, n,
                                   1.0, S, 0, 0, A.data(), n, 0.0, B.data(), d - 1),
        RandBLAS::Error);
    // lda below its lower bound (A is n-by-n).
    ASSERT_THROW(
        RandBLAS::sketch_symmetric(Layout::ColMajor, Uplo::Upper, d, n,
                                   1.0, S, 0, 0, A.data(), n - 1, 0.0, B.data(), d),
        RandBLAS::Error);
    // Submatrix window exceeding the operator: ro_s + d > S.n_rows. The
    // dense-SkOp branch has always thrown here; the sparse branch must too
    // (a silent filter would return a sketch with missing rows).
    ASSERT_THROW(
        RandBLAS::sketch_symmetric(Layout::ColMajor, Uplo::Upper, d, n,
                                   1.0, S, 1, 0, A.data(), n, 0.0, B.data(), d),
        RandBLAS::Error);
}


// MARK: LEGACY OVERLOADS

// Restores the pre-Uplo API test: the legacy sym_check_tol overloads must
// reject an asymmetric matrix at runtime, exactly as before.
TEST_F(TestSketchSymmetric, legacy_symmetry_check_fails_for_asymmetric_matrix) {
    int64_t n = 3, d = 2;
    // Column-major 3x3, symmetric except A(0,1)=5 vs A(1,0)=0.
    std::vector<double> A = {
        1.0, 0.0, 0.0,
        5.0, 2.0, 0.0,
        0.0, 0.0, 3.0
    };
    DenseDist D(d, n, ScalarDist::Uniform, Axis::Short);
    DenseSkOp<double> S(D, 42);
    RandBLAS::fill_dense(S);
    std::vector<double> B(d * n, 0.0);
    try {
        RandBLAS::sketch_symmetric(Layout::ColMajor, d, n, 1.0, S, 0, 0, A.data(), n, 0.0, B.data(), d);
        FAIL() << "Expected RandBLAS::Error for asymmetric matrix";
    } catch (const RandBLAS::Error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Symmetry check failed"), std::string::npos)
            << "Error message did not mention symmetry check: " << msg;
    }
}

// The legacy overloads (both-triangles A, runtime check, sketch_general
// forwarding) must agree with the Uplo overloads on symmetric input.
TEST_F(TestSketchSymmetric, legacy_overloads_agree_with_uplo_overloads) {
    int64_t n = 10, d = 3;
    std::vector<double> A(n * n, 0.0);
    random_symmetric_mat(n, A.data(), n, RNGState(7));
    DenseDist D(d, n, ScalarDist::Uniform);
    DenseSkOp<double> S(D, 21);
    RandBLAS::fill_dense(S);

    std::vector<double> B_legacy(d * n, 0.5), B_uplo(d * n, 0.5);
    // Left sketch, SUBMAT form, beta != 0.
    RandBLAS::sketch_symmetric(Layout::ColMajor, d, n, 2.0, S, 0, 0, A.data(), n, -1.0, B_legacy.data(), d);
    RandBLAS::sketch_symmetric(Layout::ColMajor, Uplo::Upper, d, n, 2.0, S, 0, 0, A.data(), n, -1.0, B_uplo.data(), d);
    auto msg = RandBLAS::testing::matrices_approx_equal(
        Layout::ColMajor, blas::Op::NoTrans, d, n,
        B_legacy.data(), d, B_uplo.data(), d,
        __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
        100 * std::numeric_limits<double>::epsilon(),
        10 * std::numeric_limits<double>::epsilon()
    );
    if (!msg.empty()) FAIL() << msg;

    // Right sketch, FULL form.
    DenseDist DR(n, d, ScalarDist::Uniform);
    DenseSkOp<double> SR(DR, 23);
    RandBLAS::fill_dense(SR);
    std::vector<double> C_legacy(n * d, 0.0), C_uplo(n * d, 0.0);
    RandBLAS::sketch_symmetric(Layout::ColMajor, 1.0, A.data(), n, SR, 0.0, C_legacy.data(), n);
    RandBLAS::sketch_symmetric(Layout::ColMajor, Uplo::Upper, 1.0, A.data(), n, SR, 0.0, C_uplo.data(), n);
    auto msg2 = RandBLAS::testing::matrices_approx_equal(
        Layout::ColMajor, blas::Op::NoTrans, n, d,
        C_legacy.data(), n, C_uplo.data(), n,
        __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
        100 * std::numeric_limits<double>::epsilon(),
        10 * std::numeric_limits<double>::epsilon()
    );
    if (!msg2.empty()) FAIL() << msg2;
}
