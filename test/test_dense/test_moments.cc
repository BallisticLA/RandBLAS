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
