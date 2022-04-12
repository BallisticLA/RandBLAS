#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>

#include <Random123/philox.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestDenseGaussianOp : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    static void check_normal(uint32_t seed)
    {
            int64_t n_rows = 500;
            int64_t n_cols = 500;
            int64_t size = n_rows * n_cols;
            std::vector<double> A(size);

            RandBLAS::dense_op::gen_rmat_norm<double>(n_rows, n_cols, A.data(), seed);

            double sum = 0;
            for (int i = 0; i < size; ++i)
            {
                    sum += A[i];
            }
            double mean = sum / size;

            sum = 0;
            for (int i = 0; i < size; ++i)
            {
                    sum += (A[i] - mean) * (A[i] + mean);
            }
            double stddev = std::sqrt(sum / size);

            ASSERT_NEAR(mean, 0.0, 1e-2);
            printf("Mean: %f\n", mean);
            ASSERT_NEAR(stddev, 1.0, 1e-2);
            printf("Stddev: %f\n", stddev);
    }
};

TEST_F(TestDenseGaussianOp, SimpleTest)
{
    for (uint32_t seed : {0, 1, 2})
    {
        check_normal(seed);
    }
}
