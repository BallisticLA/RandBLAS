// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/util.hh"
#include "test/comparison.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <thread>

// Fill a random matrix and truncate at the end of each row so that each row starts with a fresh counter.
template<typename T, typename RNG, typename OP>
static void fill_dense_rmat_trunc(
    T* mat,
    int64_t n_rows,
    int64_t n_cols,
    const RandBLAS::RNGState<RNG> & seed
) {

    RNG rng;
    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type k = seed.key;
    
    int ind = 0;
    int cts = n_cols / RNG::ctr_type::static_size;
    // ^ number of counters per row, where all the random numbers are to be filled in the array.
    int res = n_cols % RNG::ctr_type::static_size;
    // ^ Number of random numbers to be filled at the end of each row the the last counter of the row

    for (int i = 0; i < n_rows; i++) {
        for (int ctr = 0; ctr < cts; ctr++){
            auto rv = OP::generate(rng, c, k);
            for (int j = 0; j < RNG::ctr_type::static_size; j++) {
                mat[ind] = rv[j];
                ind++;
            }
            c.incr();
        }
        if (res != 0) { 
            for (int j = 0; j < res; j++) {
                auto rv = OP::generate(rng, c, k);
                mat[ind] = rv[j];
                ind++;
            }
            c.incr();
        }
    }
}


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

class TestDenseMoments : public ::testing::Test {
    protected:

    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_mean_stddev(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::ScalarDist sd,
        T expect_stddev
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);

        // Construct the sketching operator
        RandBLAS::DenseDist D(n_rows, n_cols, sd);
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
    auto sd = RandBLAS::ScalarDist::Gaussian;
    for (uint32_t key : {0, 1, 2})
    {
        test_mean_stddev<float>(key, 500, 500, sd, 1.0);
        test_mean_stddev<double>(key, 203, 203, sd, 1.0);
        test_mean_stddev<double>(key, 203, 503, sd, 1.0);
    }
}

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestDenseMoments, Uniform)
{
    auto sd = RandBLAS::ScalarDist::Uniform;
    double expect_stddev = 1.0;
    for (uint32_t key : {0, 1, 2})
    {
        test_mean_stddev<float>(key, 500, 500, sd, (float) expect_stddev);
        test_mean_stddev<double>(key, 203, 203, sd, expect_stddev);
        test_mean_stddev<double>(key, 203, 503, sd, expect_stddev);
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
        int stride = n_cols / 50; 
        T* mat  = new T[n_rows * n_cols];      
        T* smat = new T[n_srows * n_scols];
        fill_dense_rmat_trunc<T,RNG,OP>(mat, n_rows, n_cols, seed);
        int ind = 0; // used for indexing smat when comparing to rmat
        for (int nptr = ptr; nptr < n_cols*(n_rows-n_srows-1); nptr += stride*n_cols) {
            // ^ Loop through various pointer locations.- goes down the random matrix by amount stride.
            RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, nptr, seed);
            ind = 0;
            for (int i = 0; i<n_srows; i++) {
                // ^ Loop through entries of the submatrix
                for (int j = 0; j<n_scols; j++) {
                    ASSERT_EQ(smat[ind], mat[nptr + i*n_cols + j]);
                    ind++;
                }
            }
        }
        delete[] mat;
        delete[] smat;
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
        int stride = n_cols / 50;
        T* mat  = new T[n_rows * n_cols];      
        T* smat = new T[n_srows * n_scols];
        fill_dense_rmat_trunc<T,RNG,OP>(mat, n_rows, n_cols, seed);
        int ind = 0; // variable used for indexing smat when comparing to rmat
        for (int nptr = ptr; nptr < (n_cols - n_scols - 1); nptr += stride) {
            // ^ Loop through various pointer locations.- goes across the random matrix by amount stride.
            RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, nptr, seed);
            ind = 0;
            for (int i = 0; i<n_srows; i++) {
                // ^ Loop through entries of the submatrix
                for (int j = 0; j<n_scols; j++) {
                    ASSERT_EQ(smat[ind], mat[nptr + i*n_cols + j]);
                    ind++;
                }
            }
        }
        delete[] mat;
        delete[] smat;
    }

    template<typename T, typename RNG, typename OP>
    static void test_diag_smat_gen(
        int64_t n_cols,
        int64_t n_rows,
        const RandBLAS::RNGState<RNG> &seed
    ) {
        T* mat  = new T[n_rows * n_cols];
        T* smat = new T[n_rows * n_cols]{};
        fill_dense_rmat_trunc<T,RNG,OP>(mat, n_rows, n_cols, seed);
        int ind = 0;
        int64_t n_scols = 1;
        int64_t n_srows = 1;
        for (int ptr = 0; ptr + n_scols + n_cols*n_srows < n_cols*n_rows; ptr += n_rows+1) { // Loop through the diagonal of the matrix
            RandBLAS::util::safe_scal(n_srows * n_scols, (T) 0.0, smat, 1);
            RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, ptr, seed);
            ind = 0;
            for (int i = 0; i<n_srows; i++) { // Loop through entries of the submatrix
                for (int j = 0; j<n_scols; j++) {
                    ASSERT_EQ(smat[ind], mat[ptr + i*n_cols + j]);
                    ind++;
                }
            }
            n_scols++; // At each iteration the dimension of the submat increases
            n_srows++;
        }
        delete[] mat;
        delete[] smat;
    }

};

TEST_F(TestSubmatGeneration, col_wise)
{
    int64_t n_rows = 100;
    int64_t n_cols = 2000;
    int64_t n_srows = 41;
    int64_t n_scols = 97;
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
void DenseThreadTest(int64_t m, int64_t n) {
    int64_t d = m*n;

    std::vector<T> base(d);
    std::vector<T> test(d);

    // generate the base state with 1 thread.
    omp_set_num_threads(1);
    RandBLAS::RNGState<RNG> state(0);
    RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n, base.data(), m, n, 0, state);
    std::cerr << "with 1 thread: " << base << std::endl;

    // run with different numbers of threads, and check that the result is the same
    int n_threads = std::thread::hardware_concurrency();
    for (int i = 2; i <= n_threads; ++i) {
        std::fill(test.begin(), test.end(), (T) 0.0);
        omp_set_num_threads(i);
        RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n, test.data(), m, n, 0, state);
        std::cerr << "with " << i << " threads: " << test << std::endl;
        for (int64_t i = 0; i < d; ++i) {
            EXPECT_FLOAT_EQ( base[i], test[i] );
        }
    }
}

TEST(TestDenseThreading, UniformPhilox) {
    for (int i = 0; i < 10; ++i) {
        DenseThreadTest<float,r123::Philox4x32,r123ext::uneg11>(32, 8);
        DenseThreadTest<float,r123::Philox4x32,r123ext::uneg11>(1, 5);
        DenseThreadTest<float,r123::Philox4x32,r123ext::uneg11>(5, 1);
    }
}

TEST(TestDenseThreading, GaussianPhilox) {
    for (int i = 0; i < 10; ++i) {
        DenseThreadTest<float,r123::Philox4x32,r123ext::boxmul>(32, 8);
        DenseThreadTest<float,r123::Philox4x32,r123ext::boxmul>(1, 5);
        DenseThreadTest<float,r123::Philox4x32,r123ext::boxmul>(5, 1);
    }
}
#endif


class TestFillAxis : public::testing::Test
{
    protected:
        static inline auto distname = RandBLAS::ScalarDist::Uniform;

    template <typename T>
    static void auto_transpose(int64_t short_dim, int64_t long_dim, RandBLAS::Axis major_axis) {
        uint32_t seed = 99;
    
        // make the wide sketching operator
        RandBLAS::DenseDist D_wide(short_dim, long_dim, distname, major_axis);
        RandBLAS::DenseSkOp<T> S_wide(D_wide, seed);
        RandBLAS::fill_dense(S_wide);

        // make the tall sketching operator
        RandBLAS::DenseDist D_tall(long_dim, short_dim, distname, major_axis);
        RandBLAS::DenseSkOp<T> S_tall(D_tall, seed);
        RandBLAS::fill_dense(S_tall);

        // Sanity check: layouts are opposite.
        if (S_tall.layout == S_wide.layout) {
            FAIL() << "\n\tExpected opposite layouts.\n";
        }

        // check that buffers reflect transposed data : S_wide == S_tall.T
        auto lds_wide = (S_wide.layout == blas::Layout::ColMajor) ? short_dim : long_dim;
        auto lds_tall = (S_tall.layout == blas::Layout::ColMajor) ? long_dim  : short_dim;
        test::comparison::matrices_approx_equal(
            S_wide.layout, S_tall.layout, blas::Op::Trans, short_dim, long_dim,
            S_wide.buff, lds_wide, S_tall.buff, lds_tall,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;   
    }

};

TEST_F(TestFillAxis, autotranspose_long_axis_3x5) {
    auto_transpose<float>(3, 5, RandBLAS::Axis::Long);
}

TEST_F(TestFillAxis, autotranspose_short_axis_3x5) {
    auto_transpose<float>(3, 5, RandBLAS::Axis::Short);
}

TEST_F(TestFillAxis, autotranspose_long_axis_4x8) {
    auto_transpose<float>(4, 8, RandBLAS::Axis::Long);
}

TEST_F(TestFillAxis, autotranspose_short_axis_4x8) {
    auto_transpose<float>(4, 8, RandBLAS::Axis::Short);
}

TEST_F(TestFillAxis, autotranspose_long_axis_2x4) {
    auto_transpose<float>(2, 4, RandBLAS::Axis::Long);
}

TEST_F(TestFillAxis, autotranspose_short_axis_2x4) {
    auto_transpose<float>(2, 4, RandBLAS::Axis::Short);
}

class TestDenseSkOpStates : public ::testing::Test
{
    protected:

    template <typename T>
    static void test_concatenate_along_columns(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::ScalarDist sd
    ) {
        randblas_require(n_rows > n_cols);
        RandBLAS::DenseDist D1(    n_rows, n_cols/2,          sd, RandBLAS::Axis::Long);
        RandBLAS::DenseDist D2(    n_rows, n_cols - n_cols/2, sd, RandBLAS::Axis::Long);
        RandBLAS::DenseDist Dfull( n_rows, n_cols,            sd, RandBLAS::Axis::Long);
        RandBLAS::RNGState state(key);
        int64_t size = n_rows * n_cols;

        // Concatenates two matrices generated from state and next_state
        std::vector<T> A(size, 0.0);
        RandBLAS::DenseSkOp<T> S1(D1, state);
        RandBLAS::DenseSkOp<T> S2(D2, S1.next_state);
        RandBLAS::fill_dense(S1);
        RandBLAS::fill_dense(S2);
        int64_t size_d1 = n_rows * D1.n_cols;
        blas::copy(size_d1, S1.buff, 1, A.data(), 1);
        int64_t size_d2 = n_rows * D2.n_cols;
        blas::copy(size_d2, S2.buff, 1, A.data() + size_d1, 1);

        RandBLAS::DenseSkOp<T> S_concat(Dfull, state);
        RandBLAS::fill_dense(S_concat);

        for (int i = 0; i < size; i++) {
            ASSERT_EQ(A[i], S_concat.buff[i]);
        }
    }

    template<typename RNG>
    static void test_compute_next_state(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::ScalarDist sd
    ) {
        float *buff = new float[n_rows*n_cols];
        RandBLAS::RNGState state(key);

        RandBLAS::DenseDist D(n_rows, n_cols, sd);

        auto actual_final_state = RandBLAS::fill_dense(D, buff, state);
        auto actual_c = actual_final_state.counter;

        auto expect_final_state = RandBLAS::dense::compute_next_state(D, state);
        auto expect_c = expect_final_state.counter;

        for (int i = 0; i < RNG::ctr_type::static_size; i++) {
            ASSERT_EQ(actual_c[i], expect_c[i]);
        }

        delete [] buff;
    }
    
};

TEST_F(TestDenseSkOpStates, concat_tall_with_long_major_axis) {
    for (uint32_t key : {0, 1, 2}) {
        auto sd = RandBLAS::ScalarDist::Gaussian;
        test_concatenate_along_columns<double>(key, 13, 7, sd);
        test_concatenate_along_columns<double>(key, 80, 40, sd);
        test_concatenate_along_columns<double>(key, 83, 41, sd);
        test_concatenate_along_columns<double>(key, 91, 43, sd);
        test_concatenate_along_columns<double>(key, 97, 47, sd);
    }
}

TEST_F(TestDenseSkOpStates, compare_skopless_fill_dense_to_compute_next_state) {
    for (uint32_t key : {0, 1, 2}) {
        auto sd = RandBLAS::ScalarDist::Gaussian;
        test_compute_next_state<r123::Philox4x32>(key, 13, 7, sd);
        test_compute_next_state<r123::Philox4x32>(key, 11, 5, sd);
        test_compute_next_state<r123::Philox4x32>(key, 131, 71, sd);
        test_compute_next_state<r123::Philox4x32>(key, 80, 40, sd);
        test_compute_next_state<r123::Philox4x32>(key, 91, 43, sd);
    }
}
