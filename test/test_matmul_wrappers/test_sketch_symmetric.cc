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
#include "RandBLAS/util.hh"
#include "RandBLAS/sksy.hh"

using blas::Layout;
using blas::Uplo;
using RandBLAS::DenseDistName;
using RandBLAS::DenseDist;
using RandBLAS::DenseSkOp;
using RandBLAS::RNGState;
using RandBLAS::MajorAxis;

#include "test/comparison.hh"

#include <gtest/gtest.h>
#include <vector>


template <typename T, typename STATE>
void random_symmetric_mat(int64_t n, T* A, int64_t lda, STATE s) {
    // This function can be interpreted as first generating a random lda-by-lda symmetric matrix
    // whose entries in the upper triangle are iid, then symmetrizing that matrix, then
    // zeroing out all entries outside the leading principal submatrix of order n.
    RandBLAS::fill_dense(Layout::ColMajor, {lda, lda}, n, n, 0, 0, A, s);
    RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, A, n, lda);
    return;
}

template <typename T, typename SKOP>
blas::Side sketch_symmetric_side(
    blas::Side side_skop, blas::Layout layout, int64_t rows_out, int64_t cols_out,
    T alpha, const T* A, int64_t lda, SKOP &S, int64_t ro_s, int64_t co_s, T beta, T* B, int64_t ldb
) {
    if (side_skop == blas::Side::Left) {
        RandBLAS::sketch_symmetric(layout, rows_out, cols_out, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
        return blas::Side::Right;
    } else {
        RandBLAS::sketch_symmetric(layout, rows_out, cols_out, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
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
        uint32_t seed_a, uint32_t seed_skop, MajorAxis ma, T alpha, int64_t d, int64_t n, int64_t lda, T beta, blas::Side side_skop
    ) {
        auto [rows_out, cols_out] = dims_of_sketch_symmetric_output(d, n, side_skop);
        std::vector<T> A(lda*lda, 0.0);
        random_symmetric_mat(n, A.data(), lda, RNGState(seed_a));
        DenseDist D(rows_out, cols_out, DenseDistName::Uniform, ma);
        DenseSkOp<T> S(D, seed_skop);
        RandBLAS::fill_dense(S);
        int64_t lds = (S.layout == Layout::RowMajor) ? cols_out : rows_out;
        int64_t ldb = lds;
        uint32_t seed_b = seed_a + 42;
        std::vector<T> B_actual(d*n);
        RandBLAS::fill_dense(D, B_actual.data(), RNGState(seed_b));
        std::vector<T> B_expect(B_actual);

        // Compute the actual output
        auto side_a = sketch_symmetric_side(side_skop, S.layout, rows_out, cols_out, alpha, A.data(), lda, S, 0, 0, beta, B_actual.data(), ldb);
        // Compute the expected output
        blas::symm(S.layout, side_a, Uplo::Upper, rows_out, cols_out, alpha, A.data(), lda, S.buff, lds, beta, B_expect.data(), ldb);

        test::comparison::matrices_approx_equal(
            S.layout, blas::Op::NoTrans, rows_out, cols_out, B_actual.data(), ldb, B_expect.data(), ldb,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    }

    template <typename T>
    static void test_opposing_layouts(
        uint32_t seed_a, uint32_t seed_skop, MajorAxis ma, T alpha, int64_t d, int64_t n, int64_t lda, T beta, blas::Side side_skop
    ) {
        auto [rows_out, cols_out] = dims_of_sketch_symmetric_output(d, n, side_skop);
        std::vector<T> A(lda*lda, 0.0);
        random_symmetric_mat(n, A.data(), lda, RNGState(seed_a));
        DenseDist D(rows_out, cols_out, DenseDistName::Uniform, ma);
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
        auto side_a = sketch_symmetric_side(side_skop, layout_B, rows_out, cols_out, alpha, A.data(), lda, S, 0, 0, beta, B_actual.data(), ldb);
        // Compute the expected output
        std::vector<T> S_flipped(S.buff, S.buff + d*n);
        RandBLAS::util::flip_layout(S.layout, rows_out, cols_out, S_flipped, lds_init, ldb);
        blas::symm(layout_B, side_a, Uplo::Upper, rows_out, cols_out, alpha, A.data(), lda, S_flipped.data(), ldb, beta, B_expect.data(), ldb);

        test::comparison::matrices_approx_equal(
            layout_B, blas::Op::NoTrans, rows_out, cols_out, B_actual.data(), ldb, B_expect.data(), ldb,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    }
};


// MARK: SAME LAYOUTS

TEST_F(TestSketchSymmetric, left_sketch_10_to_3_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, left_lift_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, right_sketch_10_to_3_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
}

TEST_F(TestSketchSymmetric, right_lift_same_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_same_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_same_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
}


// MARK: OPPOSING LAYOUTS

TEST_F(TestSketchSymmetric, left_sketch_10_to_3_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, left_lift_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Left);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Left);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Left);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Left);
}

TEST_F(TestSketchSymmetric, right_sketch_10_to_3_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 3, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 3, 10, 19, -1.0, blas::Side::Right);
}


TEST_F(TestSketchSymmetric, right_lift_opposing_layouts) {
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, 0.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = 0.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, 0.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, 0.0, blas::Side::Right);
    // LDA=10,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts( 0,  1, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts( 0,  1, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 13, 10, 10, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 13, 10, 10, -1.0, blas::Side::Right);
    // LDA=19,   (seed_a, seed_skop) = (0, 1) then (31, 33),   beta = -1.0
    test_opposing_layouts(0, 1,   MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(0, 1,   MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Short, 0.5, 50, 10, 19, -1.0, blas::Side::Right);
    test_opposing_layouts(31, 33, MajorAxis::Long,  0.5, 50, 10, 19, -1.0, blas::Side::Right);
}
