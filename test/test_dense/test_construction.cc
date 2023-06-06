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

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v) {
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
        RandBLAS::DenseDistName dn,
        T expect_stddev
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);

        // Construct the sketching operator
        RandBLAS::DenseDist D = {
            .n_rows = n_rows,
            .n_cols = n_cols,
            .family = dn
        };
        auto state = RandBLAS::RNGState(key);
        auto next_state = RandBLAS::fill_dense(D, A.data(), state);

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
    auto dn = RandBLAS::DenseDistName::Gaussian;
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
    auto dn = RandBLAS::DenseDistName::Uniform;
    double expect_stddev = 1.0 / sqrt(3.0);
    for (uint32_t key : {0, 1, 2})
    {
        test_mean_stddev<float>(key, 500, 500, dn, (float) expect_stddev);
        test_mean_stddev<double>(key, 203, 203, dn, expect_stddev);
        test_mean_stddev<double>(key, 203, 503, dn, expect_stddev);
    }
}

class TestSubmatGeneration : public ::testing::Test
{

    protected:

    virtual void SetUp(){};

    virtual void TearDown(){};

    template<typename T, typename RNG, typename OP>
    static void test_colwise_smat_gen(
        int64_t n_cols,
        int64_t n_rows, 
        int64_t n_scols,
        int64_t n_srows,
        int64_t ptr,
        const RandBLAS::RNGState<RNG> &seed
    ) {
        T* mat  = new T[n_rows * n_cols];      
        T* smat = new T[n_srows * n_scols];
        RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n_cols, mat, n_rows, n_cols, 0, seed);
        int ind = 0; // used for indexing smat when comparing to rmat
        T total_error = 0;
        for (int nptr = ptr; nptr < n_cols*(n_rows-n_srows-1); nptr += n_cols) {
            // ^ Loop through various pointer locations.- goes down the random matrix
            RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, nptr, seed);
            ind = 0;
            for (int i = 0; i<n_srows; i++) {
                // ^ Loop through entries of the submatrix
                for (int j = 0; j<n_scols; j++) {
                    total_error += abs(smat[ind] - mat[nptr + i*n_cols + j]); 
                    ind++;
                }
            }
        }
        delete[] mat;
        delete[] smat;
        EXPECT_EQ(total_error, 0); //Submatrices are equal if each entry is bitwise the same
    }

    template<typename T, typename RNG, typename OP>
    static void test_rowwise_smat_gen(
        int64_t n_cols,
        int64_t n_rows, 
        int64_t n_scols,
        int64_t n_srows,
        int64_t ptr,
        const RandBLAS::RNGState<RNG> &seed
    ) {
        T* mat  = new T[n_rows * n_cols];      
        T* smat = new T[n_srows * n_scols];
        RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n_cols, mat, n_rows, n_cols, 0, seed);
        int ind = 0; // variable used for indexing smat when comparing to rmat
        T total_error = 0;
        for (int nptr = ptr; nptr < (n_cols - n_scols - 1); nptr += 1) {
            RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, nptr, seed);
            ind = 0;
            for (int i = 0; i<n_srows; i++) {
                // ^ Loop through entries of the submatrix
                for (int j = 0; j<n_scols; j++) {
                    total_error += abs(smat[ind] - mat[nptr + i*n_cols + j]);
                    ind++;
                }
            }
        }
        delete[] mat;
        delete[] smat;
        EXPECT_EQ(total_error, 0);
    }

    template<typename T, typename RNG, typename OP>
    static void test_diag_smat_gen(
        int64_t n_cols,
        int64_t n_rows,
        const RandBLAS::RNGState<RNG> &seed
    ) {
        T* mat  = new T[n_rows * n_cols];
        T* smat = new T[n_rows * n_cols]{};
        RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n_cols, mat, n_rows, n_cols, 0, seed);
        int ind = 0;
        T total_error = 0;
        int64_t n_scols = 1;
        int64_t n_srows = 1;
        for (int ptr = 0; ptr + n_scols + n_cols*n_srows < n_cols*n_rows; ptr += n_rows+1) { // Loop through the diagonal of the matrix
            RandBLAS::util::safe_scal(n_srows * n_scols, (T) 0.0, smat, 1);
            RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, ptr, seed);
            ind = 0;
            for (int i = 0; i<n_srows; i++) { // Loop through entries of the submatrix
                for (int j = 0; j<n_scols; j++) {
                    total_error += abs(smat[ind] - mat[ptr + i*n_cols + j]);
                    ind++;
                }
            }
            n_scols++; // At each iteration the dimension of the submat increases
            n_srows++;
        }
        delete[] mat;
        delete[] smat;
        EXPECT_EQ(total_error, 0);
    }
};

TEST_F(TestSubmatGeneration, col_wise)
{
    int64_t n_rows = 100;
    int64_t n_cols = 2000;
    int64_t n_srows = 40;
    int64_t n_scols = 100;
    int64_t ptr = n_rows + 2;
    for (int k = 0; k < 3; k++) {
        RandBLAS::RNGState<r123::Philox4x32> seed(k);
        test_colwise_smat_gen<float, r123::Philox4x32, r123ext::uneg11>(n_cols, n_rows, n_scols, n_srows, ptr, seed);
    }
}

TEST_F(TestSubmatGeneration, row_wise)
{
    int64_t n_rows = 100;
    int64_t n_cols = 2000;
    int64_t n_srows = 40;
    int64_t n_scols = 100;
    int64_t ptr = n_rows + 2;
    for (int k = 0; k < 3; k++) {
        RandBLAS::RNGState<r123::Philox4x32> seed(k);
        test_rowwise_smat_gen<float, r123::Philox4x32, r123ext::uneg11>(n_cols, n_rows, n_scols, n_srows, ptr, seed);
    }
}

TEST_F(TestSubmatGeneration, diag)
{
    int64_t n_rows = 100;
    int64_t n_cols = 2000;
    for (int k = 0; k < 3; k++) {
        RandBLAS::RNGState<r123::Philox4x32> seed(k);
        test_diag_smat_gen<float, r123::Philox4x32, r123ext::uneg11>(n_cols, n_rows, seed);
    }
}


#if defined(RandBLAS_HAS_OpenMP)
template <typename T, typename RNG, typename OP>
void DenseThreadTest() {
    int64_t m = 32;
    int64_t n = 8;
    int64_t d = m*n;

    std::vector<T> base(d);
    std::vector<T> test(d);

    // generate the base state with 1 thread.
    omp_set_num_threads(1);
    RandBLAS::RNGState<RNG> state(0);
    RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n, base.data(), m, n, 0, state);
    std::cerr << "with 1 thread: " << base << std::endl;

    // run with different numbers of threads, and check that the result is the same
    int n_hyper = std::thread::hardware_concurrency();
    int n_threads = std::max(n_hyper / 2, 3);

    for (int i = 2; i <= n_threads; ++i) {
        omp_set_num_threads(i);
        RandBLAS::fill_dense_submat_impl<T,RNG,OP>(n, test.data(), m, n, 0, state);
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


class TestFillAxis : public::testing::Test
{
    protected:
        static inline auto distname = RandBLAS::DenseDistName::Uniform;

    template <typename T>
    static void auto_transpose(int64_t short_dim, int64_t long_dim, RandBLAS::MajorAxis ma) {
        uint32_t seed = 99;
    
        // make the wide sketching operator
        RandBLAS::DenseDist D_wide {short_dim, long_dim, distname, ma};
        RandBLAS::DenseSkOp<T> S_wide(D_wide, seed);
        RandBLAS::fill_dense(S_wide);

        // make the tall sketching operator
        RandBLAS::DenseDist D_tall {long_dim, short_dim, distname, ma};
        RandBLAS::DenseSkOp<T> S_tall(D_tall, seed);
        RandBLAS::fill_dense(S_tall);

        // Sanity check: layouts are opposite.
        if (S_tall.layout == S_wide.layout) {
            FAIL() << "\n\tExpected opposite layouts.\n";
        }

        // check that buffers reflect transposed data : S_wide == S_tall.T
        auto lds_wide = (S_wide.layout == blas::Layout::ColMajor) ? short_dim : long_dim;
        auto lds_tall = (S_tall.layout == blas::Layout::ColMajor) ? long_dim  : short_dim;
        RandBLAS_Testing::Util::matrices_approx_equal(
            S_wide.layout, S_tall.layout, blas::Op::Trans, short_dim, long_dim,
            S_wide.buff, lds_wide, S_tall.buff, lds_tall,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;   
    }

};

TEST_F(TestFillAxis, long_axis_3x5) {
    auto_transpose<float>(3, 5, RandBLAS::MajorAxis::Long);
}

TEST_F(TestFillAxis, short_axis_3x5) {
    auto_transpose<float>(3, 5, RandBLAS::MajorAxis::Short);
}

TEST_F(TestFillAxis, long_axis_4x8) {
    auto_transpose<float>(4, 8, RandBLAS::MajorAxis::Long);
}

TEST_F(TestFillAxis, short_axis_4x8) {
    auto_transpose<float>(4, 8, RandBLAS::MajorAxis::Short);
}

TEST_F(TestFillAxis, long_axis_2x4) {
    auto_transpose<float>(2, 4, RandBLAS::MajorAxis::Long);
}

TEST_F(TestFillAxis, short_axis_2x4) {
    auto_transpose<float>(2, 4, RandBLAS::MajorAxis::Short);
}
