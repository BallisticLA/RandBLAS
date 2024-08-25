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

#include "test/test_matmul_cores/linop_common.hh"
#include <gtest/gtest.h>

using RandBLAS::SparseDist;
using RandBLAS::SparseSkOp;
using RandBLAS::MajorAxis;
using namespace test::linop_common;


class TestLSKGES : public ::testing::Test
{
    protected:
        static inline std::vector<uint32_t> keys = {42, 0, 1};
        static inline std::vector<int64_t> vec_nnzs = {1, 2, 3, 7, 19};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void apply(
        MajorAxis major_axis,
        int64_t d,
        int64_t m,
        int64_t n,
        blas::Layout layout,
        int64_t key_index,
        int64_t nnz_index,
        int threads
    ) {
        SparseDist D0(d, m, major_axis, vec_nnzs[nnz_index]);
        SparseSkOp<T> S0(D0, keys[key_index]);
        test_left_apply_to_random<T>(1.0, S0, n, 0.0, layout, threads);
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d1, // rows in sketch
        int64_t m1, // size of identity matrix
        int64_t d0, // rows in S0
        int64_t m0, // cols in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        blas::Layout layout
    ) {
        int64_t vec_nnz = d0 / 3; // this is actually quite dense. 
        SparseDist D0(d0, m0, MajorAxis::Short, vec_nnz);
        SparseSkOp<T> S0(D0, seed);
        test_left_apply_submatrix_to_eye<T>(1.0, S0, d1, m1, S_ro, S_co, layout, 0.0);
    }

    template <typename T>
    static void alpha_beta(
        uint32_t key,
        T alpha,
        T beta,
        int64_t m,
        int64_t d,
        blas::Layout layout
    ) {
        int64_t vec_nnz = d / 2;
        SparseDist DS(d, m, MajorAxis::Short, vec_nnz);
        SparseSkOp<T> S(DS, key);
        test_left_apply_submatrix_to_eye(alpha, S, d, m, 0, 0, layout, beta);
    }

    template <typename T>
    static void transpose_S(
        MajorAxis major_axis,
        uint32_t key,
        int64_t m,
        int64_t d,
        blas::Layout layout
    ) {
        randblas_require(m > d);
        bool is_saso = (major_axis == MajorAxis::Short);
        int64_t vec_nnz = (is_saso) ?  d/2 : m/2;
        SparseDist Dt(m, d, major_axis, vec_nnz);
        SparseSkOp<T> S0(Dt, key);
        test_left_apply_transpose_to_eye<T>(S0, layout);
    }

    template <typename T>
    static void submatrix_A(
        MajorAxis major_axis,
        uint32_t seed_S0, // seed for S0
        int64_t d, // rows in S0
        int64_t m, // cols in S0, and rows in A.
        int64_t n, // cols in A
        int64_t m0, // rows in A0
        int64_t n0, // cols in A0
        int64_t A_ro, // row offset for A in A0
        int64_t A_co, // column offset for A in A0
        blas::Layout layout
    ) {
        // Define the distribution for S0.
        bool is_saso = (major_axis == MajorAxis::Short);
        int64_t vec_nnz = (is_saso) ?  d/2 : m/2;
        SparseDist D(d, m, major_axis, vec_nnz);
        SparseSkOp<T> S0(D, seed_S0);
        test_left_apply_to_submatrix<T>(S0, n, m0, n0, A_ro, A_co, layout);
    }

    template <typename T>
    static void transpose_A(
        MajorAxis major_axis,
        uint32_t seed_S0, // seed for S0
        int64_t d, // rows in S0
        int64_t m, // cols in S0, and rows in A.
        int64_t n, // cols in A
        blas::Layout layout
    ) {
        // Define the distribution for S0.
        bool is_saso = (major_axis == MajorAxis::Short);
        int64_t vec_nnz = (is_saso) ?  d/2 : m/2;
        SparseDist D(d, m, major_axis, vec_nnz);
        SparseSkOp<T> S0(D, seed_S0);
        test_left_apply_to_transposed<T>(S0, n, layout);
    }
};


////////////////////////////////////////////////////////////////////////
//
//
//      Sketch with SASOs and LASOs.
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGES, sketch_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Short,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
            apply<float>(MajorAxis::Short,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
        }
    }
}

TEST_F(TestLSKGES, sketch_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Long, 19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(MajorAxis::Long, 19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

#if defined (RandBLAS_HAS_OpenMP)
TEST_F(TestLSKGES, sketch_saso_rowMajor_FourThreads)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Short,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 4
            );
            apply<float>(MajorAxis::Short,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 4
            );
        }
    }
}
#endif

TEST_F(TestLSKGES, sketch_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Short, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(MajorAxis::Short, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, sketch_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Long, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(MajorAxis::Long, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

#if defined (RandBLAS_HAS_OpenMP)
TEST_F(TestLSKGES, sketch_saso_colMajor_fourThreads)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Short, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 4);
            apply<float>(MajorAxis::Short, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 4);
        }
    }
}
#endif


////////////////////////////////////////////////////////////////////////
//
//
//      Lift with SASOs and LASOs.
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, lift_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Short,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
            apply<float>(MajorAxis::Short,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
        }
    }
}

TEST_F(TestLSKGES, lift_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Long, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(MajorAxis::Long, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, lift_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Short, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(MajorAxis::Short, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, lift_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {4, 1, 2, 3, 0}) {
            apply<double>(MajorAxis::Long, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(MajorAxis::Long, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
//
//      Submatrices of S, column major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, subset_rows_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, subset_rows_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//      Submatrices of S,row major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, subset_rows_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, subset_rows_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            1, // The first col of S is in the second col of S0
            blas::Layout::RowMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//      transpose of S
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, transpose_saso_double_colmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(MajorAxis::Short, seed, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, transpose_laso_double_colmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(MajorAxis::Long, seed, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, transpose_saso_double_rowmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(MajorAxis::Short, seed, 21, 4, blas::Layout::RowMajor);
}

TEST_F(TestLSKGES, transpose_laso_double_rowmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(MajorAxis::Long, seed, 21, 4, blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//     submatrix of A
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, saso_submatrix_a_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            MajorAxis::Short,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, saso_submatrix_a_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            MajorAxis::Short,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, laso_submatrix_a_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            MajorAxis::Long,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, laso_submatrix_a_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            MajorAxis::Long,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::RowMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//     transpose of A
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, saso_times_trans_A_colmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(MajorAxis::Short, seed, 7, 22, 5, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, laso_times_trans_A_colmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(MajorAxis::Long, seed, 7, 22, 5, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, saso_times_trans_A_rowmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(MajorAxis::Short, seed, 7, 22, 5, blas::Layout::RowMajor);
}

TEST_F(TestLSKGES, laso_times_trans_A_rowmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(MajorAxis::Long, seed, 7, 22, 5, blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//     (alpha, beta), where alpha != 1.0.
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGES, nontrivial_scales_colmajor1)
{
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, nontrivial_scales_colmajor2)
{
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, nontrivial_scales_rowmajor1)
{
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::RowMajor);
}

TEST_F(TestLSKGES, nontrivial_scales_rowmajor2)
{
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::RowMajor);
}
