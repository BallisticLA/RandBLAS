

#include <cmath>
#include <blas.hh>

#include "RandBLAS/util.hh"
#include "RandBLAS/testing/lapack_like.hh"
#include "RandBLAS/testing/comparison.hh"
#include "RandBLAS/config.h"
#include "RandBLAS/dense_skops.hh"
using RandBLAS::DenseDist;
using RandBLAS::ScalarDist;
using RandBLAS::RNGState;

#include <iostream>
#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

// MARK: Cholesky

class TestHandrolledCholesky : public ::testing::Test {
    protected:

    template <typename T, typename FUNC>
    void run_factor_gram_matrix(int n, FUNC &cholfunc, uint32_t key) {
        auto layout = blas::Layout::ColMajor;
        int64_t m = 2*n;
        DenseDist D(m, n);
        std::vector<T> A(n*n);
        std::vector<T> B(m*n);
        T iso_scale = std::pow(D.isometry_scale, 2);
        RNGState state(key);
        RandBLAS::fill_dense(D, B.data(), state);
        std::vector<T> C(B);

        // define positive definite A
        blas::syrk(layout, blas::Uplo::Upper, blas::Op::Trans, n, m, iso_scale, B.data(), m, 0.0, A.data(), n);
        RandBLAS::symmetrize(layout, blas::Uplo::Upper, n, A.data(), n);
        // overwrite A by its upper-triangular cholesky factor
        cholfunc(n, A.data());
        RandBLAS::overwrite_triangle(layout, blas::Uplo::Lower, n, 1, A.data(), n);

        // compute the gram matrix of A's cholesky factor
        blas::syrk(layout, blas::Uplo::Upper, blas::Op::Trans, n, n, 1.0, A.data(), n, 0.0, B.data(), n);
        RandBLAS::symmetrize(layout, blas::Uplo::Upper, n, B.data(), n);
        // recompute A
        blas::syrk(layout, blas::Uplo::Upper, blas::Op::Trans, n, m, iso_scale, C.data(), m, 0.0, A.data(), n);
        RandBLAS::symmetrize(layout, blas::Uplo::Upper, n, A.data(), n);

        auto msg = RandBLAS::testing::matrices_approx_equal(layout, blas::Op::NoTrans, n, n, B.data(), n, A.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
    }

    template <typename T>
    void run_sequential_factor_gram_matrix(int n, uint32_t key) {
        auto cholfunc = [](int64_t n, T* A) { 
            RandBLAS::testing::potrf_upper_sequential(n, A, n);
        };
        run_factor_gram_matrix<T>(n, cholfunc, key);
    }

    template <typename T>
    void run_blocked_factor_gram_matrix(int n, int b, uint32_t key) {
        auto cholfunc = [b](int64_t n, T* A) { 
            RandBLAS::testing::potrf_upper(n, A, n, b); 
        };
        run_factor_gram_matrix<T>(n, cholfunc, key);
    }

    template <typename T>
    void run_blocked_factor_diagonal(int n, int b) {
        auto layout = blas::Layout::ColMajor;
        std::vector<T> A(n*n, 0.0);
        std::vector<T> B(n*n);
        for (int i = 0; i < n; ++i) {
            A[i+i*n] = 1.0/((T) i + 1.0);
            B[i+i*n] = std::sqrt(A[i+i*n]);
        }
        RandBLAS::testing::potrf_upper(n, A.data(), n, b);
        auto msg = RandBLAS::testing::matrices_approx_equal(layout, blas::Op::NoTrans, n, n, B.data(), n, A.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
    }

};

TEST_F(TestHandrolledCholesky, sequential_random_gram_matrix) {
    for (int i = 1; i <= 64; ++i) {
        run_sequential_factor_gram_matrix<float>(i, 0);
    }
    for (int i = 1; i <= 64; ++i) {
        run_sequential_factor_gram_matrix<float>(i, 1);
    }
}

TEST_F(TestHandrolledCholesky, blocked_diagonal) {
    for (int i = 2; i <= 64; i+=2) {
        run_blocked_factor_diagonal<float>(i, 2);
    }
    for (int i = 2; i <= 64; i+=2) {
        run_blocked_factor_diagonal<float>(i, 3);
    }
    for (int i = 2; i <= 256; i*=2) {
        run_blocked_factor_diagonal<float>(i, i/2);
    }
    for (int i = 16; i <= 256; i*=2) {
        run_blocked_factor_diagonal<float>(i, i-7);
    }
    for (int i = 16; i <= 256; i*=2) {
        run_blocked_factor_diagonal<float>(i-3, i/4);
    }
}

TEST_F(TestHandrolledCholesky, blocked_random_gram_matrix) {
    for (int i = 2; i <= 16; i+=2) {
        run_blocked_factor_gram_matrix<float>(i, 1, 0);
    }
    for (int i = 2; i <= 64; i+=2) {
        run_blocked_factor_gram_matrix<float>(i, 2, 0);
    }
    for (int i = 2; i <= 96; i*=2) {
        run_blocked_factor_gram_matrix<float>(i, i/2, 0);
    }
    for (int i = 16; i <= 96; i*=2) {
        run_blocked_factor_gram_matrix<float>(i, i-5, 0);
    }
    for (int i = 16; i <= 96; i*=2) {
        run_blocked_factor_gram_matrix<float>(i-5, 7, 0);
    }
}



template <typename T>
void verify_product(int64_t m, int64_t n, int64_t k, const T* A, const T* B, const T* C) {
    using std::vector;
    int64_t lda = m;
    int64_t ldb = k;
    int64_t ldc = m;
    vector<T> C_work(m*n, 0.0);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, (T) 1.0, A, lda, B, ldb, (T) 0.0, C_work.data(), ldc);

    vector<T> A_copy(m*k, 0.0);
    vector<T> B_copy(k*n, 0.0);
    blas::copy(m*k, A, 1, A_copy.data(), 1);
    blas::copy(k*n, B, 1, B_copy.data(), 1);
    std::for_each(A_copy.begin(), A_copy.end(), [](T &val) {val = std::abs(val); });
    std::for_each(B_copy.begin(), B_copy.end(), [](T &val) {val = std::abs(val); });
    vector<T> E(m*n, 0.0);
    T err_alpha = 2 * k * std::numeric_limits<T>::epsilon();
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, err_alpha, A_copy.data(), lda, B_copy.data(), ldb, (T)0.0, E.data(), m);
    auto msg = RandBLAS::testing::buffs_approx_equal(C_work.data(), C, E.data(), m*n, __PRETTY_FUNCTION__, __FILE__, __LINE__);
    if (msg.size() > 0) {
        FAIL() << msg;
    }
    return;
}

template <typename T>
void verify_orthonormal_componentwise(int m, int n, T* Q, T max_cond = 10) {
    std::vector<T> I(n*n, 0.0);
    for (int i = 0; i < n; ++i)
        I[i + i*n] = 1.0;
    std::vector<T> QtQ(n*n, 0.0);
    T tol = max_cond * std::sqrt(m) * std::numeric_limits<T>::epsilon();
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, n, n, m, (T) 1.0, Q, m, Q, m, (T) 0.0, QtQ.data(), n);
    auto msg = RandBLAS::testing::matrices_approx_equal(blas::Layout::ColMajor, blas::Op::NoTrans, n, n, QtQ.data(), n, I.data(), n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__, tol, tol
    );
    if (msg.size() > 0) {
        FAIL() << msg;
    }
    return;
}

// MARK: QR

class TestHandrolledQR : public ::testing::Test {
    protected:

    template <typename T>
    void run_cholqr_gaussian(int m, int n, int b, uint32_t key) {
        DenseDist D(m, n, ScalarDist::Gaussian);
        std::vector<T> A(m*n);
        T iso_scale = D.isometry_scale;
        RNGState state(key);
        RandBLAS::fill_dense(D, A.data(), state);
        blas::scal(m*n, iso_scale, A.data(), 1);

        std::vector<T> Q(A);
        std::vector<T> R(2*n*n);
        RandBLAS::testing::chol_qr(m, n, Q.data(), R.data(), b, true);
        verify_orthonormal_componentwise(m, n, Q.data());
        verify_product(m, n, n, Q.data(), R.data(), A.data());
    }

};

TEST_F(TestHandrolledQR, cholqr_small) {
    for (uint32_t key = 111; key < 112; ++key) {
        run_cholqr_gaussian<float>(10, 1, 5, key);
        run_cholqr_gaussian<float>(2,  2, 5, key);
        run_cholqr_gaussian<float>(10, 6, 5, key);

        run_cholqr_gaussian<float>(10, 1, 2, key);
        run_cholqr_gaussian<float>(10, 2, 2, key);
        run_cholqr_gaussian<float>(10, 6, 2, key);
    }
}

TEST_F(TestHandrolledQR, cholqr_medium) {
    for (uint32_t key = 111; key < 112; ++key) {
        run_cholqr_gaussian<float>(1000, 100, 5,  key);
        run_cholqr_gaussian<float>(1024, 128, 64, key);
    }
}


// MARK: Eigenvalues

template <typename T>
std::vector<T> posdef_with_random_eigvecs(std::vector<T> &eigvals, uint32_t key) {
    int64_t n = eigvals.size();
    for (auto ev : eigvals) 
        randblas_require(ev > 0);
    std::vector<T> vecs(n*n);
    DenseDist distn(n, n, ScalarDist::Gaussian);
    RNGState state(key);
    RandBLAS::fill_dense(distn, vecs.data(), state);
    std::vector<T> Q(n*n, 0.0);
    for (int i = 0; i < n; ++i)
        Q[i + i*n] = 1.0;
    RandBLAS::testing::apply_reflectors(n, n, n, vecs.data(), n, Q.data(), n);
    for (int i = 0; i < n; ++i)
        blas::scal(n, std::sqrt(eigvals[i]), Q.data() + i*n, 1);
    std::vector<T> out(n*n, 0.0);
    blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, n, n, (T)1.0, Q.data(), n, (T)0.0, out.data(), n);
    RandBLAS::symmetrize(blas::Layout::ColMajor, blas::Uplo::Upper, n, out.data(), n);
    return out;
}

class TestHandrolledEigvals : public ::testing::Test {
    protected:

    template <typename T>
    void run_diag(int n, int b) {
        std::vector<T> A(n*n, 0.0);
        std::vector<T> eigvals_expect(n);
        for (int i = 0; i < n; ++i) {
            eigvals_expect[i] = 1.0 / (1 + (T)i);
            A[i + i*n] = eigvals_expect[i];
        }
        std::vector<T> eigvals_actual(n);
        int64_t iters = 1;
        T tol = 1e-3;
        auto iter = RandBLAS::testing::posdef_eig_chol_iteration(n, A.data(), eigvals_actual.data(), tol, iters, b);
        ASSERT_EQ(iter, 0);
        auto msg = RandBLAS::testing::buffs_approx_equal(eigvals_actual.data(), eigvals_expect.data(), n, __PRETTY_FUNCTION__, __FILE__, __LINE__,  tol, tol);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    template <typename T>
    void run_general_posdef(int n, int b, uint32_t key) {
        std::vector<T> eigvals_expect(n);
        for (int i = 0; i < n; ++i) {
            eigvals_expect[i] = 2.0 / (1 + (T)i);
        }
        auto A = posdef_with_random_eigvecs(eigvals_expect, key);
        std::vector<T> eigvals_actual(n);
        int64_t iters = 1000;
        T tol = 1e-3;
        auto iter = RandBLAS::testing::posdef_eig_chol_iteration(n, A.data(), eigvals_actual.data(), tol, iters, b);
        std::cout << "Number of iterations  : " << iter << std::endl;
        T min_eig_actual = *std::min_element(eigvals_actual.begin(), eigvals_actual.end());
        T max_eig_actual = *std::max_element(eigvals_actual.begin(), eigvals_actual.end());
        std::cout << "min_comp / min_actual " << min_eig_actual / eigvals_expect[n-1] << std::endl;
        std::cout << "max_comp / max_actual " << max_eig_actual / eigvals_expect[0] << std::endl;
        std::string msg;
        msg = RandBLAS::testing::approx_equal(min_eig_actual, eigvals_expect[n-1], __PRETTY_FUNCTION__, __FILE__, __LINE__,  tol, tol);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::approx_equal(max_eig_actual, eigvals_expect[0  ], __PRETTY_FUNCTION__, __FILE__, __LINE__,  tol, tol);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    template <typename T>
    void run_power_general_posdef(int n, uint32_t key) {
        std::vector<T> eigvals_expect(n);
        for (int i = 0; i < n; ++i) {
            eigvals_expect[i] = 2.0 / std::sqrt((1 + (T)i));
        }
        auto A = posdef_with_random_eigvecs(eigvals_expect, key);

        std::vector<T> ourwork(2*n, 0.0);
        std::vector<T> subwork{};
        T tol = 1e-3;
        RNGState state(key + 1);
        T p_fail = (n < 500) ? (T) 1e-6 : (T) 0.5;
        // ^ This affects the number of iterations that will be used in the power method. That number is
        //   max{#iterations to succeed in expectation, #iterations to succeed with some probability}.
        //   Large values of p_fail (like p_fail = 0.5) just say "succeed in expectation."
        auto [lambda_max, lambda_min, ignore] = RandBLAS::testing::exeigs_powermethod(
            n, A.data(), ourwork.data(), tol, p_fail, state, subwork
        );

        std::cout << "min_comp / min_actual = " << lambda_min / eigvals_expect[n-1] << std::endl;
        std::cout << "max_comp / max_actual = " << lambda_max / eigvals_expect[0] << std::endl;
        std::string msg;
        msg = RandBLAS::testing::approx_equal(lambda_min, eigvals_expect[n-1], __PRETTY_FUNCTION__, __FILE__, __LINE__,  tol, tol);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::approx_equal(lambda_max, eigvals_expect[0  ], __PRETTY_FUNCTION__, __FILE__, __LINE__,  tol, tol);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }
};

TEST_F(TestHandrolledEigvals, power_smallish) {
    run_power_general_posdef<double>(10, 0);
    run_power_general_posdef<double>(50, 0);
    run_power_general_posdef<double>(100, 0);
}

TEST_F(TestHandrolledEigvals, diag) {
    run_diag<double>(5, 1);
    run_diag<double>(10, 1);
    run_diag<double>(100, 1);
}

TEST_F(TestHandrolledEigvals, power_medium) {
    run_power_general_posdef<double>( 512, 0 );
    run_power_general_posdef<double>( 512, 1 );
}

TEST_F(TestHandrolledEigvals, power_largeish) {
    run_power_general_posdef<float>( 1024, 0 );
    run_power_general_posdef<float>( 1024, 1 );
}


