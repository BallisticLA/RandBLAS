#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <thread>


class TestDenseMoments : public ::testing::Test
{
    protected:

    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_mean_stddev(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::dense::DenseDistName dn,
        T expect_stddev
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);

        // Construct the sketching operator
        RandBLAS::dense::DenseDist D = {
            .n_rows = n_rows,
            .n_cols = n_cols,
            .family = dn
        };
        auto state = RandBLAS::base::RNGState(key);
        auto next_state = RandBLAS::dense::fill_buff(A.data(), D, state);

        // Compute the entrywise empirical mean and standard deviation.
        T mean = std::accumulate(A.data(), A.data() + size, 0.0) /size;
        T sum = 0;
        std::for_each(A.data(), A.data() + size, [&] (T elem) {
            sum += (elem - mean) * (elem - mean);
        });
        T stddev = std::sqrt(sum / (size - 1));

        // We're only interested in mean-zero random variables.
        // Standard deviation depends on the distribution.
        EXPECT_NEAR(mean, 0.0, 1e-2) << "Initial state:\n\t" << state << "\nFinal state:\n\t" << next_state;
        EXPECT_NEAR(stddev, expect_stddev, 1e-2);
    }
};

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseMoments, Gaussian)
{
    auto dn = RandBLAS::dense::DenseDistName::Gaussian;
    for (uint32_t key : {0, 1, 2})
    {
        test_mean_stddev<float>(key, 500, 500, dn, 1.0);
        test_mean_stddev<double>(key, 203, 203, dn, 1.0);
        test_mean_stddev<double>(key, 203, 503, dn, 1.0);
    }
}

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseMoments, Uniform)
{
    auto dn = RandBLAS::dense::DenseDistName::Uniform;
    double expect_stddev = 1.0 / sqrt(3.0);
    for (uint32_t key : {0, 1, 2})
    {
        test_mean_stddev<float>(key, 500, 500, dn, (float) expect_stddev);
        test_mean_stddev<double>(key, 203, 203, dn, expect_stddev);
        test_mean_stddev<double>(key, 203, 503, dn, expect_stddev);
    }
}


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
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, NULL, layout);
        if (preallocate)
            RandBLAS::dense::realize_full(S0);
        RandBLAS::dense::DenseSkOp<T> S0_ref(D, seed, NULL, layout);
        RandBLAS::dense::realize_full(S0_ref);

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
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, 0, A.data(), lda,
            0.0, B.data(), ldb
        );

        // check the result
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), S0_ref.buff, d*m,
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
        RandBLAS::dense::DenseSkOp<T> S0(Dt, seed, NULL, layout);
        RandBLAS::dense::realize_full(S0);
        bool is_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
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
        RandBLAS_Testing::Util::matrices_approx_equal(
            S0.layout, blas::Op::Trans, d, m,
            B.data(), ldb, S0.buff, lds,
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
        bool is_colmajor = layout == blas::Layout::ColMajor;
        int64_t pos = (is_colmajor) ? (S_ro + d0 * S_co) : (S_ro * m0 + S_co);
        assert(d0 * m0 >= pos + d * m);

        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = d0,
            .n_cols = m0,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, NULL,  layout);
        RandBLAS::dense::realize_full(S0);
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
            S_ptr, lds,
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
        RandBLAS::dense::DenseSkOp<T> S0(D, seed_S0, NULL, layout);
        RandBLAS::dense::realize_full(S0);
        bool is_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A0(m0 * n0, 0.0);
        uint32_t seed_A0 = 42000;
        RandBLAS::dense::DenseDist DA0 = {.n_rows = m0, .n_cols = n0};
        RandBLAS::dense::fill_buff(A0.data(), DA0, RandBLAS::base::RNGState(seed_A0));
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




template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v)
{
    size_t n = v.size();
    os << "{";
    if (n)
    {
        os << v[0];
        for (size_t i = 1; i < n; ++i)
            os << ", " << v[i];
    }
    os << "}";
    return os;
}

#if defined(RandBLAS_HAS_OpenMP)
template <typename T, typename RNG, typename OP>
void DenseThreadTest()
{
    int64_t m = 32;
    int64_t n = 8;
    int64_t d = m*n;

    std::vector<T> base(d);
    std::vector<T> test(d);

    // generate the base state with 1 thread.
    omp_set_num_threads(1);
    RandBLAS::dense::fill_rmat<T,RNG,OP>(m, n, base.data(), {});
    std::cerr << "with 1 thread: " << base << std::endl;

    // run with different numbers of threads, and check that the result is the same
    int n_hyper = std::thread::hardware_concurrency();
    int n_threads = std::max(n_hyper / 2, 3);

    for (int i = 2; i <= n_threads; ++i) {

        omp_set_num_threads(i);

        RandBLAS::dense::fill_rmat<T,RNG,OP>(m, n, test.data(), {});

        std::cerr << "with " << i << " threads: " << test << std::endl;

        for (int64_t i = 0; i < d; ++i) {
            EXPECT_FLOAT_EQ( base[i], test[i] );
        }
    }
}

TEST(TestDenseThreading, UniformPhilox) {
    DenseThreadTest<float,r123::Philox4x32,r123ext::uneg11>();
}

TEST(TestDenseThreading, GaussianPhilox) {
    DenseThreadTest<float,r123::Philox4x32,r123ext::boxmul>();
}

#endif




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
        blas::Layout layout
    ) {
        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = m,
            .n_cols = d,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };

        // Define the sketching operator struct, S0.
        // Create a copy that we always realize explicitly.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, NULL, layout);
        if (preallocate)
            RandBLAS::dense::realize_full(S0);
        RandBLAS::dense::DenseSkOp<T> S0_ref(D, seed, NULL, layout);
        RandBLAS::dense::realize_full(S0_ref);

        // define a matrix to be sketched, and create workspace for sketch.
        bool is_colmajor = layout == blas::Layout::ColMajor;
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t lda = m;
        int64_t ldb = (is_colmajor) ? m : d;

        // Perform the sketch
        RandBLAS::dense::rskge3<T>(
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            m, d, m,
            1.0, A.data(), lda,
            S0, 0, 0,
            0.0, B.data(), ldb
        );

        // check the result
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), S0_ref.buff, d*m,
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
            .n_rows = d,
            .n_cols = m,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(Dt, seed, NULL, layout);
        RandBLAS::dense::realize_full(S0);
        bool is_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        int64_t ldb = (is_colmajor) ? m : d;

        // perform the sketch
        RandBLAS::dense::rskge3<T>(
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::Trans,
            m, d, m,
            1.0, A.data(), m,
            S0, 0, 0,
            0.0, B.data(), ldb   
        );

        // check that B == S.T
        int64_t lds = (is_colmajor) ? d : m;
        RandBLAS_Testing::Util::matrices_approx_equal(
            S0.layout, blas::Op::Trans, m, d,
            B.data(), ldb, S0.buff, lds,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    // TODO: enable tests below for submatrix of S and submatrix of A.

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d, // columns in sketch
        int64_t m, // size of identity matrix
        int64_t d0, // cols in S0
        int64_t m0, // rows in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        blas::Layout layout
    ) {
        assert(d0 > d);
        assert(m0 > m);
        bool is_colmajor = layout == blas::Layout::ColMajor;
        int64_t pos = (is_colmajor) ? (S_ro + m0 * S_co) : (S_ro * d0 + S_co);
        assert(d0 * m0 >= pos + d * m);

        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = m0,
            .n_cols = d0,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed, NULL,  layout);
        RandBLAS::dense::realize_full(S0);
        int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(m * d, 0.0);
        int64_t lda = m;
        int64_t ldb = (is_colmajor) ? m : d;
        
        // Perform the sketch
        RandBLAS::dense::rskge3<T>(
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            m, d, m,
            1.0, A.data(), lda,
            S0, S_ro, S_co,
            0.0, B.data(), ldb   
        );
        // Check the result
        T *S_ptr = &S0.buff[pos];
        RandBLAS_Testing::Util::matrices_approx_equal(
            S0.layout, blas::Op::NoTrans,
            m, d,
            B.data(), ldb,
            S_ptr, lds,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
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
        blas::Layout layout
    ) {
        assert(m0 > m);
        assert(n0 > n);

        // Define the distribution for S0.
        RandBLAS::dense::DenseDist D = {
            .n_rows = n,
            .n_cols = d,
            .family = RandBLAS::dense::DenseDistName::Gaussian
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense::DenseSkOp<T> S0(D, seed_S0, NULL, layout);
        RandBLAS::dense::realize_full(S0);
        bool is_colmajor = layout == blas::Layout::ColMajor;

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A0(m0 * n0, 0.0);
        uint32_t seed_A0 = 42000;
        RandBLAS::dense::DenseDist DA0 = {.n_rows = m0, .n_cols = n0};
        RandBLAS::dense::fill_buff(A0.data(), DA0, RandBLAS::base::RNGState(seed_A0));
        std::vector<T> B(m * d, 0.0);
        int64_t lda = (is_colmajor) ? DA0.n_rows : DA0.n_cols;
        int64_t ldb = (is_colmajor) ? m : d;
        
        // Perform the sketch
        int64_t a_offset = (is_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
        T *A_ptr = &A0.data()[a_offset]; 
        RandBLAS::dense::rskge3<T>(
            S0.layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            m, d, n,
            1.0, A_ptr, lda,
            S0, 0, 0,
            0.0, B.data(), ldb   
        );

        // Check the result
        int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;
        std::vector<T> B_expect(m * d, 0.0);
        blas::gemm<T>(S0.layout, blas::Op::NoTrans, blas::Op::NoTrans,
            m, d, n,
            1.0, A_ptr, lda, S0.buff, lds,
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
//      RSKGE3: Basic sketching (vary preallocation, row vs col major)
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGE3, right_sketch_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::RowMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::RowMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_single_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_sketch_eye_single_null)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false, blas::Layout::ColMajor);
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
        sketch_eye<double>(seed, 10, 51, true, blas::Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_lift_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, blas::Layout::RowMajor);
}

TEST_F(TestRSKGE3, right_lift_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, blas::Layout::ColMajor);
}

TEST_F(TestRSKGE3, right_lift_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, blas::Layout::RowMajor);
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
        transpose_S<double>(seed, 200, 30, blas::Layout::ColMajor);
}

TEST_F(TestRSKGE3, transpose_double_rowmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, blas::Layout::RowMajor);
}

TEST_F(TestRSKGE3, transpose_single)
{
    for (uint32_t seed : {0})
        transpose_S<float>(seed, 200, 30, blas::Layout::ColMajor);
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
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGE3, submatrix_s_double_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGE3, submatrix_s_single) 
{
    for (uint32_t seed : {0})
        submatrix_S<float>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
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
            blas::Layout::ColMajor
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
            blas::Layout::RowMajor
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
            blas::Layout::ColMajor
        );
}
