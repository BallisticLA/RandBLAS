#include <RandBLAS/dense.hh>
#include <RandBLAS/util.hh>
#include <rbtutil.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <numeric>
#include <Random123/philox.h>


class TestDenseMoments : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_mean_stddev(
        uint32_t seed,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::dense::DenseDistName dn,
        T expect_stddev
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);

        // Construct the sketching operator
        RandBLAS::dense::DenseDist D {
            .family = dn,
            .n_rows = n_rows,
            .n_cols = n_cols
        };
        RandBLAS::dense::fill_buff<T>(A.data(), D, RandBLAS::base::RNGState{seed, 0});

        // Compute the entrywise empirical mean and standard deviation.
        T mean = std::accumulate(A.data(), A.data() + size, 0.0) /size;
        T sum = 0;
        std::for_each(A.data(), A.data() + size, [&] (T elem) {
            sum += (elem - mean) * (elem - mean);
        });
        T stddev = std::sqrt(sum / (size - 1));

        // We're only interested in mean-zero random variables.
        // Standard deviation depends on the distribution.
        EXPECT_NEAR(mean, 0.0, 1e-2);
        EXPECT_NEAR(stddev, expect_stddev, 1e-2);
    }
};

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseMoments, Gaussian)
{
    auto dn = RandBLAS::dense::DenseDistName::Gaussian;
    for (uint32_t seed : {0, 1, 2})
    {
        test_mean_stddev<float>(seed, 500, 500, dn, 1.0);
        test_mean_stddev<double>(seed, 203, 203, dn, 1.0);
        test_mean_stddev<double>(seed, 203, 503, dn, 1.0);
    }
}

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseMoments, Uniform)
{
    auto dn = RandBLAS::dense::DenseDistName::Uniform;
    double expect_stddev = 1.0 / sqrt(3.0);
    for (uint32_t seed : {0, 1, 2})
    {
        test_mean_stddev<float>(seed, 500, 500, dn, (float) expect_stddev);
        test_mean_stddev<double>(seed, 203, 203, dn, expect_stddev);
        test_mean_stddev<double>(seed, 203, 503, dn, expect_stddev);
    }
}


class TestLSKGE3 : public ::testing::Test
{
    /*
    Things that need to be tested:
        1. Sketching the identity matrix works as expected.
        2. Transposing the sketching operator works as expected.
        3. Using a submatrix of the sketching operator.
        4. sketching a submatrix of a data matrix.
        5. row-major sketching
            5.1 An exception should be raised if S0.layout != 
                declared layout.

    Things that don't need to be tested:
        ?
    */
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
            .family = RandBLAS::dense::DenseDistName::Gaussian,
            .n_rows = d,
            .n_cols = m
        };

        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> *S0_ptr;
        std::vector<T> buff; // awkward.
        if (preallocate) {
            buff.resize(d * m, 0.0);
            S0_ptr = new RandBLAS::dense::DenseSkOp<T>(
                D, seed, 0, buff.data(), false, true, layout
            );
        } else {
            S0_ptr = new RandBLAS::dense::DenseSkOp<T>(
                D, seed, 0, buff.data(), false, true, layout
            );
        }

        // define a matrix to be sketched, and create workspace for sketch.
        bool is_colmajor = layout == blas::Layout::ColMajor;
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t lda = m;
        int64_t ldb = (is_colmajor) ? d : m;

        // Perform the sketch
        RandBLAS::dense::lskge3<T>(
            (*S0_ptr).layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, *S0_ptr, 0, 0, A.data(), lda,
            0.0, B.data(), ldb
        );

        // check the result
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), (*S0_ptr).buff, d*m);
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
            .family=RandBLAS::dense::DenseDistName::Gaussian,
            .n_rows = m,
            .n_cols = d
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(Dt, seed, 0,
            NULL, false, true, layout
        );
         bool is_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t lda = m;
        int64_t ldb = (is_colmajor) ? d : m;

        // perform the sketch
        RandBLAS::dense::lskge3<T>(
            S0.layout,
            blas::Op::Trans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, 0, A.data(), m,
            0.0, B.data(), ldb   
        );

        // check that B == S.T
        int64_t lds = (is_colmajor) ? m : d;
        RandBLAS_Testing::Util::matrices_approx_equal<T>(
            S0.layout, blas::Op::Trans, d, m,
            B.data(), ldb, S0.buff, lds      
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
        bool is_colmajor = layout == blas::Layout::ColMajor;
        int64_t pos = (is_colmajor) ? (S_ro + d0 * S_co) : (S_ro * m0 + S_co);
        assert(d0 * m0 >= pos + d * m);

        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .family = RandBLAS::dense::DenseDistName::Gaussian,
            .n_rows = d0,
            .n_cols = m0
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, 0,
            NULL, false, true, layout
        );
        int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t lda = m;
        int64_t ldb = (is_colmajor) ? d : m;
        
        // Perform the sketch
        RandBLAS::dense::lskge3<T>(
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, S_ro, S_co,
            A.data(), lda,
            0.0, B.data(), ldb   
        );
        // Check the result
        T *S_ptr = &S0.buff[pos];
        RandBLAS_Testing::Util::matrices_approx_equal(
            S0.layout, blas::Op::NoTrans,
            d, m,
            B.data(), ldb,
            S_ptr, lds
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
            .family = RandBLAS::dense::DenseDistName::Gaussian,
            .n_rows = d,
            .n_cols = m
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed_S0, 0,
            NULL, false, true, layout
        );
        bool is_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A0(m0 * n0, 0.0);
        uint32_t ctr_A0 = 42;
        uint32_t seed_A0 = 42000;
        RandBLAS::dense::DenseDist DA0 = {.n_rows = m0, .n_cols = n0};
        RandBLAS::dense::fill_buff(A0.data(), DA0, RandBLAS::base::RNGState{ctr_A0, seed_A0});
        std::vector<T> B(d * n, 0.0);
        int64_t lda = (is_colmajor) ? DA0.n_rows : DA0.n_cols;
        int64_t ldb = (is_colmajor) ? d : n;
        
        // Perform the sketch
        int64_t a_offset = (is_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
        T *A_ptr = &A0.data()[a_offset]; 
        RandBLAS::dense::lskge3<T>(
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, n, m,
            1.0, S0, 0, 0,
            A_ptr, lda,
            0.0, B.data(), ldb   
        );

        // Check the result
        int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;
        std::vector<T> B_expect(d * n, 0.0);
        blas::gemm<T>(S0.layout, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S0.buff, lds, A_ptr, lda,
            0.0, B_expect.data(), ldb
        );
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), B_expect.data(), d * n);
    }

};

TEST_F(TestLSKGE3, eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::RowMajor);
}

TEST_F(TestLSKGE3, eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::RowMajor);
}

TEST_F(TestLSKGE3, eye_single_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKGE3, eye_single_null)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false, blas::Layout::ColMajor);
}

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
