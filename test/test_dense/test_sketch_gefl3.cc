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
        // Define the distribution for S0, and S0 itself.
        // Create a copy that we always realize explicitly.
        RandBLAS::DenseDist D(d, m);
        RandBLAS::DenseSkOp<T> S0(D, seed, nullptr);
        if (preallocate)
            RandBLAS::fill_dense(S0);
        RandBLAS::DenseSkOp<T> S0_ref(D, seed, nullptr);
        RandBLAS::fill_dense(S0_ref);

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
        // Define the distribution for S0, and S0 itself.
        RandBLAS::DenseDist Dt(m, d);
        RandBLAS::DenseSkOp<T> S0(Dt, seed, nullptr);
        RandBLAS::DenseSkOp<T> S0_ref(Dt, seed, nullptr);
        RandBLAS::fill_dense(S0_ref);

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

        // Define the distribution for S0, and S0 itself.
        RandBLAS::DenseDist D(d0, m0);
        RandBLAS::DenseSkOp<T> S0(D, seed, nullptr);
        RandBLAS::DenseSkOp<T> S0_ref(D, seed, nullptr);
        RandBLAS::fill_dense(S0_ref);
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

        // Define the distribution for S0, and S0 itself.
        RandBLAS::DenseDist D(d, m);
        RandBLAS::DenseSkOp<T> S0(D, seed_S0, nullptr);
        RandBLAS::fill_dense(S0);
        bool AB_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A0(m0 * n0, 0.0);
        uint32_t seed_A0 = 42000;
        RandBLAS::DenseDist DA0(m0, n0);
        RandBLAS::fill_dense(DA0, A0.data(), RandBLAS::RNGState(seed_A0));
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
        blas::Op opS = (S0.layout == layout) ? blas::Op::NoTrans : blas::Op::Trans;
        blas::gemm<T>(layout, opS, blas::Op::NoTrans,
            d, n, m,
            1.0, S0.buff, lds, A_ptr, lda,
            0.0, B_expect.data(), ldb
        );
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), B_expect.data(), d * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template<typename T>
    static void test_sketch_vec_wide(
        uint32_t seed,
        int64_t d,
        int64_t m,
        int64_t incx,
        int64_t incy
    ) {
        T total = 0;
        T x[incx*m];
        T y_wide[incy*d];
        T y_ref[incy*d];
    
        for (int i = 0; i < incx*m; i++){
            x[i] = 1.0;
        }

        RandBLAS::DenseDist D(d, m);

        RandBLAS::DenseSkOp<double> S_wide(D, seed, nullptr);
        RandBLAS::fill_dense(S_wide);

        RandBLAS::sketch_vector<T>(blas::Op::NoTrans, d, m, 1, S_wide, 0, 0, x, incx, 0, y_wide, incy);

        //Possible: repalce blas::Layout::Rowmajor with S0.layout
        int64_t lds = (S_wide.layout == blas::Layout::RowMajor) ? m : d;
        blas::gemv(S_wide.layout, blas::Op::NoTrans, d, m, 1, S_wide.buff, lds, x, incx, 0, y_ref, incy); 

        for (int i = 0; i < d; i++) {
            total += abs(y_wide[incy*i] - y_ref[incy*i]);
        }

        ASSERT_EQ(total, 0);
    }
    
    template<typename T>
    static void test_sketch_vec_tall(
        uint32_t seed,
        int64_t d,
        int64_t m,
        int64_t incx,
        int64_t incy
    ) {
        T total = 0;
        T x[incx*m];
        T y_tall[incy*d];
        T y_ref[incy*d];
    
        for (int i = 0; i < incx*m; i++){
            x[i] = 1.0;
        }

        RandBLAS::DenseDist D_tall(m, d);

        RandBLAS::DenseSkOp<double> S_tall(D_tall, seed, nullptr);
        RandBLAS::fill_dense(S_tall);


        RandBLAS::sketch_vector<T>(blas::Op::Trans, d, m, 1, S_tall, 0, 0, x, incx, 0, y_tall, incy);

        //Possible: repalce blas::Layout::Rowmajor with S0.layout
        int64_t lds = (S_tall.layout == blas::Layout::RowMajor) ? d : m;
        blas::gemv(S_tall.layout, blas::Op::Trans, m, d, 1, S_tall.buff, lds, x, incx, 0, y_ref, incy); 

        for (int i = 0; i < d; i++) {
            total += abs(y_tall[incy*i] - y_ref[incy*i]);
        }

        ASSERT_EQ(total, 0);
    }

};

//TODO: Write testing functions that generates a tall sketching operator with OpS = Trans. Should be the same as generating a wide operator oith no trans.
//      Test tall sketching operators.    
TEST_F(TestLSKGE3, test_sketch_vec_wide)
{
    for (uint32_t seed : {0, 1, 2}) {
        test_sketch_vec_wide<double>(seed, 1000, 10000, 2, 3);
        test_sketch_vec_wide<double>(seed, 997, 17371, 3, 2);
    }
}

TEST_F(TestLSKGE3, test_sketch_vec_tall)
{
    for (uint32_t seed : {0, 1, 2}) {
        test_sketch_vec_tall<double>(seed, 1000, 10000, 2, 3);
        test_sketch_vec_tall<double>(seed, 997, 17371, 3, 2);
    }
}
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

