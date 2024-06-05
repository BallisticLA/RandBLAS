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

#include "../linop_common.hh"
#include <gtest/gtest.h>

using namespace test::linop_common;
using RandBLAS::DenseDist;
using RandBLAS::DenseSkOp;
using blas::Layout;

class TestRSKGE3 : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void sketch_eye(
        uint32_t seed,
        int64_t m,
        int64_t d,
        bool preallocate,
        Layout layout
    ) {
        DenseDist D(m, d);
        DenseSkOp<T> S0(D, seed, nullptr);
        if (preallocate)
            RandBLAS::fill_dense(S0);
        test_right_apply_submatrix_to_eye<T>(1.0, S0, m, d, 0, 0, layout, 0.0, 0);
    }

    template <typename T>
    static void transpose_S(
        uint32_t seed,
        int64_t m,
        int64_t d,
        Layout layout
    ) {
        DenseDist Dt(d, m);
        DenseSkOp<T> S0(Dt, seed, nullptr);
        test_right_apply_tranpose_to_eye<T>(S0, layout);
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d, // columns in sketch
        int64_t m, // size of identity matrix
        int64_t d0, // cols in S0
        int64_t m0, // rows in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        Layout layout
    ) {
        DenseDist D(m0, d0);
        DenseSkOp<T> S0(D, seed);
        test_right_apply_submatrix_to_eye<T>(1.0, S0, m, d, S_ro, S_co, layout, 0.0, 0);
    }

    template <typename T>
    static void submatrix_A(
        uint32_t seed_S0, // seed for S
        int64_t d, // cols in S
        int64_t m, // rows in A.
        int64_t n, // cols in A, rows in S.
        int64_t m0, // rows in A0
        int64_t n0, // cols in A0
        int64_t A_ro, // row offset for A in A0
        int64_t A_co, // column offset for A in A0
        Layout layout
    ) {
        DenseDist D(n, d);
        DenseSkOp<T> S0(D, seed_S0, nullptr);
        test_right_apply_to_submatrix<T>(S0, m, m0, n0, A_ro, A_co, layout);
    }

};


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGE3: Basic sketching (vary preallocation, row vs col major)
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGE3, right_sketch_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, Layout::RowMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, Layout::RowMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_single_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true, Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_single_null)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false, Layout::ColMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGE3: Lifting
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGE3, right_lift_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_lift_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, Layout::RowMajor);
}

TEST_F(TestRSKGE3, right_lift_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_lift_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGE3: transpose of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGE3, transpose_double_colmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, Layout::ColMajor);
}

TEST_F(TestRSKGE3, transpose_double_rowmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, Layout::RowMajor);
}

TEST_F(TestRSKGE3, transpose_single)
{
    for (uint32_t seed : {0})
        transpose_S<float>(seed, 200, 30, Layout::ColMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      RSKGE3: Submatrices of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGE3, submatrix_s_double_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            2, // The first row of S is in the third row of S0
            1, // The first col of S is in the second col of S0
            Layout::ColMajor
        );
}

TEST_F(TestRSKGE3, submatrix_s_double_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            2, // The first row of S is in the third row of S0
            1, // The first col of S is in the second col of S0
            Layout::RowMajor
        );
}

TEST_F(TestRSKGE3, submatrix_s_single) 
{
    for (uint32_t seed : {0})
        submatrix_S<float>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            2, // The first row of S is in the third row of S0
            1, // The first col of S is in the second col of S0
            Layout::ColMajor
        );
}

////////////////////////////////////////////////////////////////////////
//
//
//     RSKGE3: submatrix of A
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGE3, submatrix_a_double_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(seed,
            3, // number of columns in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            Layout::ColMajor
        );
}

TEST_F(TestRSKGE3, submatrix_a_double_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(seed,
            3, // number of columns in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            Layout::RowMajor
        );
}

TEST_F(TestRSKGE3, submatrix_a_single) 
{
    for (uint32_t seed : {0})
        submatrix_A<float>(seed,
            3, // number of columns in sketch.
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            Layout::ColMajor
        );
}
