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

        RandBLAS::dense_op::gen_rmat_norm<T>(n_rows, n_cols, A.data(), seed, 0);

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
        //RandBLAS::util::genmat(n_rows, n_cols, A.data(), seed);
        RandBLAS::dense_op::gen_rmat_unif<T>(n_rows, n_cols, A.data(), seed, 0);

        T mean = std::accumulate(A.data(), A.data() + size, 0.0) / size;

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


class TestLSKGE3 : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void dummy_run_gaussian(uint32_t seed, int64_t m, int64_t n, int64_t d, bool preallocate)
    {
        // Define the sketching operator struct, S0.
        RandBLAS::dense_op::SketchingBuffer<T> S0 = {
            .dist=RandBLAS::dense_op::DenseDist::Gaussian,
            .ctr_offset=0,
            .key=seed,
            .n_rows=d,
            .n_cols=m,
            .op_data=NULL,
            .populated=false,
            .persistent=false
        };
        if (preallocate) {
            std::vector<T> buff(d * m, 0.0);
            S0.op_data = buff.data();
            S0.populated = false;
            S0.persistent = true;
        }

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * n, 0.0);
        RandBLAS::dense_op::gen_rmat_unif<T>(m, n, A.data(), seed + 1, 0);
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