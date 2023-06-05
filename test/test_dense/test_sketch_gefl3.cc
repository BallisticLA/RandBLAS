#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"
#include "RandBLAS/skge.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <thread>


class TestLSKGE3 : public ::testing::Test
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
        blas::Layout layout
    ) {
        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = d,
            .n_cols = m,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };

        // Define the sketching operator struct, S0.
        // Create a copy that we always realize explicitly.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, nullptr);
        if (preallocate)
            RandBLAS::dense::fill_dense(S0);
        RandBLAS::dense::DenseSkOp<T> S0_ref(D, seed, nullptr);
        RandBLAS::dense::fill_dense(S0_ref);

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t lda = m;
        int64_t ldb = (layout == blas::Layout::ColMajor) ? d : m;

        // Perform the sketch
        RandBLAS::sketch_general<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, 0, A.data(), lda,
            0.0, B.data(), ldb
        );

        // check the result
        int64_t lds = (S0.layout == blas::Layout::ColMajor) ? S0.dist.n_rows : S0.dist.n_cols;
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, S0.layout, blas::Op::NoTrans, d, m, B.data(), ldb, S0_ref.buff, lds,
                __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void transpose_S(
        uint32_t seed,
        int64_t m,
        int64_t d,
        blas::Layout layout
    ) {
        // Define the distribution for S0.
        RandBLAS::dense::DenseDist Dt = {
            .n_rows = m,
            .n_cols = d,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(Dt, seed, nullptr);
        RandBLAS::dense::DenseSkOp<T> S0_ref(Dt, seed, nullptr);
        RandBLAS::dense::fill_dense(S0_ref);

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t ldb = (layout == blas::Layout::ColMajor) ? d : m;

        // perform the sketch
        RandBLAS::sketch_general<T>(
            layout,
            blas::Op::Trans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, 0, A.data(), m,
            0.0, B.data(), ldb   
        );

        // check that B == S.T
        int64_t lds = (S0.layout == blas::Layout::ColMajor) ? S0.dist.n_rows : S0.dist.n_cols;
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, S0.layout, blas::Op::Trans, d, m, B.data(), ldb, S0_ref.buff, lds,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d, // rows in sketch
        int64_t m, // size of identity matrix
        int64_t d0, // rows in S0
        int64_t m0, // cols in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        blas::Layout layout
    ) {
        assert(d0 > d);
        assert(m0 > m);

        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = d0,
            .n_cols = m0,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };

        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, nullptr);
        RandBLAS::dense::DenseSkOp<T> S0_ref(D, seed, nullptr);
        RandBLAS::dense::fill_dense(S0_ref);
        bool S0_colmajor = S0.layout == blas::Layout::ColMajor;
        int64_t lds = (S0_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;
        int64_t pos = (S0_colmajor) ? (S_ro + lds * S_co) : (S_ro * lds + S_co);
        assert(d0 * m0 >= pos + d * m);

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t lda = m;
        int64_t ldb = (layout == blas::Layout::ColMajor) ? d : m;
        
        // Perform the sketch
        RandBLAS::sketch_general<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, S_ro, S_co,
            A.data(), lda,
            0.0, B.data(), ldb   
        );

        // Check the result
        T *S_ptr = &S0_ref.buff[pos];
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, S0.layout, blas::Op::NoTrans, d, m, B.data(), ldb, S_ptr, lds,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void submatrix_A(
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
        assert(m0 > m);
        assert(n0 > n);

        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = d,
            .n_cols = m,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed_S0, nullptr);
        RandBLAS::dense::fill_dense(S0);
        bool AB_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A0(m0 * n0, 0.0);
        uint32_t seed_A0 = 42000;
        RandBLAS::dense::DenseDist DA0 = {.n_rows = m0, .n_cols = n0};
        RandBLAS::dense::fill_dense(DA0, A0.data(), RandBLAS::base::RNGState(seed_A0));
        std::vector<T> B(d * n, 0.0);
        int64_t lda = (AB_colmajor) ? DA0.n_rows : DA0.n_cols;
        int64_t ldb = (AB_colmajor) ? d : n;
        
        // Perform the sketch
        int64_t a_offset = (AB_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
        T *A_ptr = &A0.data()[a_offset]; 
        RandBLAS::sketch_general<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, n, m,
            1.0, S0, 0, 0,
            A_ptr, lda,
            0.0, B.data(), ldb   
        );

        // Check the result
        int64_t lds = (S0.layout == blas::Layout::ColMajor) ? S0.dist.n_rows : S0.dist.n_cols;
        std::vector<T> B_expect(d * n, 0.0);
        blas::Op transS = (S0.layout == layout) ? blas::Op::NoTrans : blas::Op::Trans;
        blas::gemm<T>(layout, transS, blas::Op::NoTrans,
            d, n, m,
            1.0, S0.buff, lds, A_ptr, lda,
            0.0, B_expect.data(), ldb
        );
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), B_expect.data(), d * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

};


////////////////////////////////////////////////////////////////////////
//
//
//      Basic sketching (vary preallocation, row vs col major)
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGE3, sketch_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, sketch_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::RowMajor);
}

TEST_F(TestLSKGE3, sketch_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, sketch_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::RowMajor);
}

TEST_F(TestLSKGE3, sketch_eye_single_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, sketch_eye_single_null)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false, blas::Layout::ColMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      Lifting
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGE3, lift_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, lift_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, blas::Layout::RowMajor);
}

TEST_F(TestLSKGE3, lift_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, lift_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      transpose of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGE3, transpose_double_colmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, transpose_double_rowmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, blas::Layout::RowMajor);
}

TEST_F(TestLSKGE3, transpose_single)
{
    for (uint32_t seed : {0})
        transpose_S<float>(seed, 200, 30, blas::Layout::ColMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      Submatrices of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGE3, submatrix_s_double_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 12, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGE3, submatrix_s_double_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 12, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGE3, submatrix_s_single) 
{
    for (uint32_t seed : {0})
        submatrix_S<float>(seed,
            3, 10, // (rows, cols) in S.
            8, 12, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}

////////////////////////////////////////////////////////////////////////
//
//
//     submatrix of A
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGE3, submatrix_a_double_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGE3, submatrix_a_double_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGE3, submatrix_a_single) 
{
    for (uint32_t seed : {0})
        submatrix_A<float>(seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::ColMajor
        );
}

