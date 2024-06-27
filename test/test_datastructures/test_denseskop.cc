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
        RandBLAS::DenseDistName dn,
        T expect_stddev
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);

        // Construct the sketching operator
        RandBLAS::DenseDist D(n_rows, n_cols, dn);
        auto state = RandBLAS::RNGState(key);
        auto [layout, next_state] = RandBLAS::fill_dense(D, A.data(), state);

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
        int stride = n_cols / 50; 
        T* mat  = new T[n_rows * n_cols];      
        T* smat = new T[n_srows * n_scols];
        fill_dense_rmat_trunc<T,RNG,OP>(mat, n_rows, n_cols, seed);
        int ind = 0; // used for indexing smat when comparing to rmat
        T total_error = 0;
        for (int nptr = ptr; nptr < n_cols*(n_rows-n_srows-1); nptr += stride*n_cols) {
            // ^ Loop through various pointer locations.- goes down the random matrix by amount stride.
            RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, nptr, seed);
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
        int stride = n_cols / 50;
        T* mat  = new T[n_rows * n_cols];      
        T* smat = new T[n_srows * n_scols];
        fill_dense_rmat_trunc<T,RNG,OP>(mat, n_rows, n_cols, seed);
        int ind = 0; // variable used for indexing smat when comparing to rmat
        T total_error = 0;
        for (int nptr = ptr; nptr < (n_cols - n_scols - 1); nptr += stride) {
            // ^ Loop through various pointer locations.- goes across the random matrix by amount stride.
            RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, nptr, seed);
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
        fill_dense_rmat_trunc<T,RNG,OP>(mat, n_rows, n_cols, seed);
        int ind = 0;
        T total_error = 0;
        int64_t n_scols = 1;
        int64_t n_srows = 1;
        for (int ptr = 0; ptr + n_scols + n_cols*n_srows < n_cols*n_rows; ptr += n_rows+1) { // Loop through the diagonal of the matrix
            RandBLAS::util::safe_scal(n_srows * n_scols, (T) 0.0, smat, 1);
            RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n_cols, smat, n_srows, n_scols, ptr, seed);
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
void DenseThreadTest() {
    int64_t m = 32;
    int64_t n = 8;
    int64_t d = m*n;

    std::vector<T> base(d);
    std::vector<T> test(d);

    // generate the base state with 1 thread.
    omp_set_num_threads(1);
    RandBLAS::RNGState<RNG> state(0);
    RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n, base.data(), m, n, 0, state);
    std::cerr << "with 1 thread: " << base << std::endl;

    // run with different numbers of threads, and check that the result is the same
    int n_hyper = std::thread::hardware_concurrency();
    int n_threads = std::max(n_hyper / 2, 3);

    for (int i = 2; i <= n_threads; ++i) {
        omp_set_num_threads(i);
        RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(n, test.data(), m, n, 0, state);
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
        test::comparison::matrices_approx_equal(
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


class TestStateUpdate : public ::testing::Test
{
    protected:

    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_var_mat_gen(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::DenseDistName dn
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);
        std::vector<T> B(size, 0.0);

        // Construct the sketching operator
        RandBLAS::DenseDist D(n_rows, n_cols, dn);

        auto state = RandBLAS::RNGState(key);
        auto [layout, next_state] = RandBLAS::fill_dense(D, A.data(), state);
        RandBLAS::fill_dense(D, B.data(), next_state);

        ASSERT_TRUE(!(A == B));
    }

    template <typename T>
    static void test_identity(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::DenseDistName dn
    ) {
        // Allocate workspace
        int64_t size = n_rows * n_cols;
        std::vector<T> A(size, 0.0);
        std::vector<T> B(size, 0.0);

        double total = 0.0;

        // Construct the sketching operator
        RandBLAS::DenseDist D1(
            n_rows,
            n_cols / 2,
            dn
        );

        RandBLAS::DenseDist D3(
            n_rows,
            n_cols - n_cols / 2,
            dn
        );

        RandBLAS::DenseDist D2(
            n_rows,
            n_cols,
            dn
        );

        auto state = RandBLAS::RNGState(key);
        auto state1 = RandBLAS::RNGState(key);

        // Concatenates two matrices generated from state and next_state
        auto [layout, next_state] = RandBLAS::fill_dense(D1, A.data(), state);
        RandBLAS::fill_dense(D3, A.data() + (int64_t) (D1.n_rows * D1.n_cols), next_state);

        RandBLAS::fill_dense(D2, B.data(), state1);

        for (int i = 0; i < size; i++) {
            total += abs(A[i] - B[i]);
        }
        
        ASSERT_TRUE(total == 0.0);
    }

    template<typename RNG>
    static void test_finalstate(
        uint32_t key,
        int64_t n_rows,
        int64_t n_cols,
        RandBLAS::DenseDistName dn
    ) {
        int total = 0;
        int *buff = new int[n_rows*n_cols];
        auto state = RandBLAS::RNGState(key);
        auto state_copy = RandBLAS::RNGState(key);

        RandBLAS::DenseDist D(n_rows, n_cols, dn);

        typename RNG::ctr_type c_ref = state_copy.counter;

        auto [layout, final_state] = RandBLAS::fill_dense(D, buff, state);
        auto c = final_state.counter;
        int c_len = c.size();

        int64_t pad = 0; //Pad computed such that  n_cols+pad is divisible by RNG::static_size
        if (n_rows % RNG::ctr_type::static_size != 0) {
            pad = RNG::ctr_type::static_size - n_rows % RNG::ctr_type::static_size;
        }
        int64_t last_ptr = n_rows-1 + (n_cols-1)*n_rows;
        int64_t last_ptr_padded = last_ptr + last_ptr/n_rows * pad;
        c_ref.incr(last_ptr_padded / RNG::ctr_type::static_size + 1); 

        for (int i = 0; i < c_len; i++) {
            total += c[i] - c_ref[i];
        }

        ASSERT_TRUE(total == 0);
        delete [] buff;
    }
    
};

// For small matrix sizes, mean and stddev are not very close to desired vals.
TEST_F(TestStateUpdate, Gaussian_var_gen)
{
    for (uint32_t key : {0, 1, 2}) {
        auto dn = RandBLAS::DenseDistName::Gaussian;
        test_var_mat_gen<double>(key, 100, 50, dn);
    }
}

TEST_F(TestStateUpdate, Gaussian_identity)
{
    for (uint32_t key : {0, 1, 2}) {
        auto dn = RandBLAS::DenseDistName::Gaussian;
        test_identity<double>(key, 13, 7, dn);
        test_identity<double>(key, 80, 40, dn);
        test_identity<double>(key, 83, 41, dn);
        test_identity<double>(key, 91, 43, dn);
        test_identity<double>(key, 97, 47, dn);
    }
}

TEST_F(TestStateUpdate, Final_State)
{
    for (uint32_t key : {0, 1, 2}) {
        auto dn = RandBLAS::DenseDistName::Gaussian;
        test_finalstate<r123::Philox4x32>(key, 13, 7, dn);
        test_finalstate<r123::Philox4x32>(key, 11, 5, dn);
        test_finalstate<r123::Philox4x32>(key, 131, 71, dn);
        test_finalstate<r123::Philox4x32>(key, 80, 40, dn);
        test_finalstate<r123::Philox4x32>(key, 91, 43, dn);
    }
}
