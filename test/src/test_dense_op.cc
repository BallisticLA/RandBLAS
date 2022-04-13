#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>

#include <Random123/philox.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;


/*
Need more tests for dense operators - any ideas?
*/
class TestDenseGaussianOp : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void check_normal(uint32_t seed, int64_t n_rows, int64_t n_cols)
    {
            int64_t size = n_rows * n_cols;
            std::vector<T> A(size, 0.0);

            RandBLAS::dense_op::gen_rmat_norm<T>(n_rows, n_cols, A.data(), seed);

            T sum = 0;
            for (int i = 0; i < size; ++i)
            {
                    sum += A[i];
            }
            T mean = sum / size;

            sum = 0;
            for (int j = 0; j < size; ++j)
            {
                    T elem = A[j];
                    sum += (elem - mean) * (elem - mean);
            }
            T stddev = std::sqrt(sum / (size - 1));


            printf("Mean: %f\n", mean);
            printf("Stddev: %f\n", stddev);

            ASSERT_NEAR(mean, 0.0, 1e-2);
            ASSERT_NEAR(stddev, 1.0, 1e-2);
    }
};

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseGaussianOp, SimpleTest)
{
    for (uint32_t seed : {0, 1, 2})
    {
        check_normal<float>(seed, 500, 500);
        check_normal<double>(seed, 203, 203);

        check_normal<double>(seed, 203, 503);
    }
}
