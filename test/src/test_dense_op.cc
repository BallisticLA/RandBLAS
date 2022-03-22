#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>



void main() {


}


static void check_normal() {
    int64_t n_rows = 500;
    int64_t n_cols = 500;
    int64_t size = n_rows* n_cols;
    std::vector<float> A(size);
    uint64_t seed = 12;
    // Error here - Something's off with namespace visibility
    //RandBLAS::dense_op::gen_rmat_normal(n_rows, n_cols, A.data(), seed);

    int sum = 0;
    for (int i = 0; i < size; ++i)
    {
            sum += A[i];
    }
    float mean = sum / size;

    sum = 0;
    for (int i = 0; i < size; ++i)
    {
            sum += (A[i] - mean) * (A[i] + mean);
    }
    float stddev = std::sqrt(sum / size);

    printf("Mean: %d", mean);
    printf("Stddev: %d", stddev);
}