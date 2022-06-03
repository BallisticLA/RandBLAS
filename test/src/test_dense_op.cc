#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <numeric>

#include <Random123/philox.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestDenseGaussianOp : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    // Compares mean and stddev to the expected ones
    static void test_normal_m_std(uint32_t seed, int64_t n_rows, int64_t n_cols)
    {
            int64_t size = n_rows * n_cols;
            std::vector<T> A(size, 0.0);

            RandBLAS::dense_op::gen_rmat_norm<T>(n_rows, n_cols, A.data(), seed);

            T mean = std::accumulate(A.data(), A.data() + size, 0.0) /size;

            T sum = 0;
            std::for_each(A.data(), A.data() + size, [&] (T elem) {
                sum += (elem - mean) * (elem - mean);
            });
            T stddev = std::sqrt(sum / (size - 1));

            printf("Mean: %f\n", mean);
            printf("Stddev: %f\n", stddev);

            // We expect this from a set of gaussian random numbers 
            ASSERT_NEAR(mean, 0.0, 1e-2);
            ASSERT_NEAR(stddev, 1.0, 1e-2);
    }
};

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseGaussianOp, SimpleTest)
{
    for (uint32_t seed : {0, 1, 2})
    {
        test_normal_m_std<float>(seed, 500, 500);
        test_normal_m_std<double>(seed, 203, 203);
        test_normal_m_std<double>(seed, 203, 503);
    }
}

class TestDenseUniformOp : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    // Compares mean and stddev to the expected ones
    static void test_unif_m_std(uint32_t seed, int64_t n_rows, int64_t n_cols)
    {
            int64_t size = n_rows * n_cols;
            std::vector<T> A(size, 0.0);

            // Uniform {-1, 1}, never 0.
            RandBLAS::util::genmat(n_rows, n_cols, A.data(), seed);
            //RandBLAS::dense_op::gen_rmat_unif<T>(n_rows, n_cols, A.data(), seed);

            T mean = std::accumulate(A.data(), A.data() + size, 0.0) /size;

            T sum = 0;
            std::for_each(A.data(), A.data() + size, [&] (T elem) {
                sum += (elem - mean) * (elem - mean);
            });
            T stddev = std::sqrt(sum / (size - 1));

            printf("Mean: %f\n", mean);
            printf("Stddev: %f\n", stddev);

            // We expect this from a set of iid uniformly distributed random numbers .
            // Mean should be close to 0, since we have values in the interval {-1, 1}.
            ASSERT_NEAR(mean, 0.0, 1e-2);
            // Compare to the result of a standard formula for a uniform distribution with the given {-1, 1}.
            ASSERT_NEAR(stddev, sqrt(4.0 / 12.0), 1e-2);
    }
};

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseUniformOp, SimpleTest)
{
    for (uint32_t seed : {0, 1, 2})
    {
        test_unif_m_std<float>(seed, 500, 500);
        test_unif_m_std<double>(seed, 203, 203);
        test_unif_m_std<double>(seed, 203, 503);
    }
}
