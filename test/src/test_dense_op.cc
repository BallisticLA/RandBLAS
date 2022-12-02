#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <numeric>

#include <Random123/philox.h>

#define RELTOL_POWER 0.7
#define ABSTOL_POWER 0.75

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
        RandBLAS::dense_op::DistName dn,
        T expect_stddev
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);

        // Construct the sketching operator
        RandBLAS::dense_op::Dist D {
            .n_rows = n_rows,
            .n_cols = n_cols,
            .family = dn
        };
        RandBLAS::dense_op::fill_buff<T>(A.data(), D, seed, 0);

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
    auto dn = RandBLAS::dense_op::DistName::Gaussian;
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
    auto dn = RandBLAS::dense_op::DistName::Uniform;
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
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void sketch_eye(
        uint32_t seed,
        int64_t m,
        int64_t d,
        bool preallocate
    ) {
        // Define the distribution for S0.
        RandBLAS::dense_op::Dist D = {
            .family = RandBLAS::dense_op::DistName::Gaussian,
            .n_rows = d,
            .n_cols = m
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense_op::SketchingOperator<T> S0 = {
            .dist = D,
            .ctr_offset = 0,
            .key = seed,
            .op_data = NULL,
            .filled = false,
            .persistent = true,
            .layout = blas::Layout::ColMajor
        };
        if (preallocate) {
            std::vector<T> buff(d * m, 0.0);
            S0.op_data = buff.data();
            S0.filled = false;
            S0.persistent = true;
        }

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        T *A_ptr = A.data();
        for (int i = 0; i < m; ++i)
            A_ptr[i + m*i] = 1.0;


        // Perform the sketch
        std::vector<T> B(d * m, 0.0);
        RandBLAS::dense_op::lskge3<T>(
            blas::Layout::ColMajor,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, A_ptr, m,
            0.0, B.data(), d   
        );

        // check the result
        T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
        T *S0_ptr = S0.op_data;
        for (int64_t i = 0; i < m*d; ++i) {
            T actual = B[i];
            T expect = S0_ptr[i];
            T atol = reltol * std::min(abs(actual), abs(expect));
            EXPECT_NEAR(actual, expect, atol);
        }
    }

    template <typename T>
    static void transpose(
        uint32_t seed,
        int64_t m,
        int64_t d
    ) {
        // Define the distribution for S0.
        RandBLAS::dense_op::Dist Dt = {
            .family=RandBLAS::dense_op::DistName::Gaussian,
            .n_rows = m,
            .n_cols = d
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense_op::SketchingOperator<T> S0 = {
            .dist=Dt,
            .ctr_offset=0,
            .key=seed,
            .persistent = true,
            .layout=blas::Layout::ColMajor
        };

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        T *A_ptr = A.data();
        for (int i = 0; i < m; ++i)
            A_ptr[i + m*i] = 1.0;


        // Perform the sketch
        std::vector<T> B(d * m, 0.0);
        RandBLAS::dense_op::lskge3<T>(
            blas::Layout::ColMajor,
            blas::Op::Trans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, A_ptr, m,
            0.0, B.data(), d   
        );

        // check that B == S.T
        T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
        T *S0_ptr = S0.op_data;
        for (int64_t i = 0; i < d; ++i) {
            for (int64_t j = 0; j < m; ++j) {
                T actual = B[i + d*j];
                T expect = S0_ptr[j + m*i];
                T atol = reltol * std::min(abs(actual), abs(expect));
                EXPECT_NEAR(actual, expect, atol);
            }
        }
    }
};

TEST_F(TestLSKGE3, eye_double_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true);
}

TEST_F(TestLSKGE3, eye_double_null)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false);
}

TEST_F(TestLSKGE3, eye_single_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true);
}

TEST_F(TestLSKGE3, eye_single_null)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false);
}

TEST_F(TestLSKGE3, transpose_double)
{
    for (uint32_t seed : {0})
        transpose<double>(seed, 200, 30);
}

TEST_F(TestLSKGE3, transpose_single)
{
    for (uint32_t seed : {0})
        transpose<float>(seed, 200, 30);
}
