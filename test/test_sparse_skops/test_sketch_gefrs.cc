#include "../common.hh"
#include "RandBLAS/skge.hh"
#include <gtest/gtest.h>
#include <math.h>


class TestRSKGES : public ::testing::Test
{
    protected:
        static inline std::vector<uint32_t> keys = {42, 0, 1};
        static inline std::vector<int64_t> vec_nnzs = {1, 2, 3, 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void sketch_eye(
        uint32_t seed,
        int64_t m,
        int64_t d,
        RandBLAS::MajorAxis major_axis,
        int64_t vec_nnz,
        blas::Layout layout
    ) {
        RandBLAS::SparseDist D = {
            .n_rows = m,
            .n_cols = d,
            .vec_nnz = vec_nnz,
            .major_axis = major_axis
        };
        RandBLAS::SparseSkOp<T> S0(D, seed);
        RandBLAS::fill_sparse(S0);
        test::common::test_right_apply_submatrix_to_eye<T>(1.0, S0, m, d, 0, 0, layout, 0.0, 0);
    }

    template <typename T>
    static void apply(
        RandBLAS::MajorAxis major_axis,
        int64_t d,
        int64_t m,
        int64_t n,
        blas::Layout layout,
        int64_t key_index,
        int64_t nnz_index,
        int threads
    ) {
        RandBLAS::SparseDist D = {
            .n_rows=n, .n_cols=d, .vec_nnz=vec_nnzs[nnz_index], .major_axis=major_axis
        };
        RandBLAS::SparseSkOp<T> S0(D, keys[key_index]);
        RandBLAS::fill_sparse(S0);
        test::common::test_right_apply_to_random<T>(1.0, S0, m, layout, 0.0, threads);
        

    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d1, // cols in sketch
        int64_t n1, // size of identity matrix
        int64_t d0, // cols in S0
        int64_t n0, // rows in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        blas::Layout layout
    ) {
        randblas_require(d0 >= d1);
        randblas_require(n0 >= n1);
        int64_t vec_nnz = d0 / 3; // this is actually quite dense. 
        RandBLAS::SparseSkOp<T> S0(
            {n0, d0, vec_nnz, RandBLAS::MajorAxis::Short}, seed
        );
        RandBLAS::fill_sparse(S0);
        test::common::test_right_apply_submatrix_to_eye<T>(1.0, S0, n1, d1, S_ro, S_co, layout, 0.0, 0);
    }
};

////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Basic sketching
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGES, right_sketch_eye_saso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::MajorAxis::Short, 1, blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_sketch_eye_saso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::MajorAxis::Short, 1,  blas::Layout::RowMajor);
}

TEST_F(TestRSKGES, right_sketch_eye_laso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::MajorAxis::Long, 1,  blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_sketch_eye_laso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::MajorAxis::Long, 1,  blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Lifting
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGES, right_lift_eye_saso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::MajorAxis::Short, 5, blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_lift_eye_saso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::MajorAxis::Short, 5, blas::Layout::RowMajor);
}

TEST_F(TestRSKGES, right_lift_eye_laso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::MajorAxis::Long, 13, blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_lift_eye_laso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::MajorAxis::Long, 13, blas::Layout::RowMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: more sketching
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGES, sketch_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Short,
                12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
            apply<float>(RandBLAS::MajorAxis::Short,
                12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
        }
    }
}


TEST_F(TestRSKGES, sketch_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Long, 12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::MajorAxis::Long, 12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, sketch_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Short, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::MajorAxis::Short, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, sketch_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Long, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::MajorAxis::Long, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: more lifting
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRSKGES, lift_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Short,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
            apply<float>(RandBLAS::MajorAxis::Short,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
        }
    }
}

TEST_F(TestRSKGES, lift_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Long, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::MajorAxis::Long, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, lift_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Short, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::MajorAxis::Short, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, lift_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::MajorAxis::Long, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::MajorAxis::Long, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Submatrices of S, column major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRSKGES, subset_rows_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGES, subset_rows_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            3,  // The first column of S is in the forth column of S0
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            1, // The first row of S is in the second row of S0
            0,
            blas::Layout::ColMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Submatrices of S,row major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRSKGES, subset_rows_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGES, subset_rows_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            3, // The first column of S is in the forth column of S0
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            1, // The first row of S is in the second row of S0
            0,
            blas::Layout::RowMajor
        );
}
