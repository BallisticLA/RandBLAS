#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <numeric>

#include <Random123/philox.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

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
        RandBLAS::dense_op::fill_buff_iid<T>(A.data(), D, seed, 0);

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
    static void dummy_run_gaussian(uint32_t seed, int64_t m, int64_t n, int64_t d, bool preallocate)
    {
        // Define the distribution for S0.
        RandBLAS::dense_op::Dist D = {
            .family=RandBLAS::dense_op::DistName::Gaussian,
            .n_rows = d,
            .n_cols = n
        };
        // Define the sketching operator struct, S0.
        RandBLAS::dense_op::SketchingOperator<T> S0 = {
            .dist=D,
            .ctr_offset=0,
            .key=seed,
            .op_data=NULL,
            .filled=false,
            .persistent=false,
            .layout=blas::Layout::ColMajor
        };
        if (preallocate) {
            std::vector<T> buff(d * m, 0.0);
            S0.op_data = buff.data();
            S0.filled = false;
            S0.persistent = true;
        }

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * n, 0.0);
        RandBLAS::util::genmat<T>(m, n, A.data(), seed + 1);
        std::vector<T> B(d * n, 0.0);

        // Perform the sketch
        RandBLAS::dense_op::lskge3<T>(
            blas::Layout::ColMajor,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, n, m,
            1.0,
            S0, 0,
            A.data(), m,
            0.0,
            B.data(), d   
        );
        // ... Check that nothing insane happened.
    }
};

TEST_F(TestLSKGE3, TrivialTest_Double)
{
    for (uint32_t seed : {0})
    {
        dummy_run_gaussian<double>(seed, 200, 10, 30, false);
        dummy_run_gaussian<double>(seed, 200, 10, 30, true);
    }
}

TEST_F(TestLSKGE3, TrivialTest_Single)
{
    for (uint32_t seed : {0})
    {
        dummy_run_gaussian<float>(seed, 200, 10, 30, false);
        dummy_run_gaussian<float>(seed, 200, 10, 30, true);
    }
}