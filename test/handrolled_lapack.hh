
#pragma once

#include "RandBLAS/util.hh"
#include "RandBLAS/dense_skops.hh"

#include <blas.hh>
#include <iostream>
#include <vector>
#include <cmath>

namespace hr_lapack {

using RandBLAS::RNGState;

template <typename T>
int potrf_upper_sequential(int64_t n, T* A, int64_t lda) {
    // Cache access is much better if the matrix is lower triangular.
    // Could implement as lower triangular and then call transpose_square.
    for (int64_t j = 0; j < n; ++j) {
        if (A[j + j * lda] <= 0) {
            std::cout << "Cholesky failed at index " << (j+1) << " of " << n << ".";
            return j+1;
        }
        A[j + j * lda] = std::sqrt(A[j + j * lda]);
        for (int64_t i = j + 1; i < n; ++i) {
            A[j + i * lda] /= A[j + j * lda];
        }
        for (int64_t k = j + 1; k < n; ++k) {
            for (int64_t i = k; i < n; ++i) {
                A[k + i * lda] -= A[j + i * lda] * A[j + k * lda];
            }
        }
    }
    return 0;
}

template <typename T>
int potrf_upper(int64_t n, T* A, int64_t lda, int64_t b = 64) {
    randblas_require(b > 0);
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    int64_t curr_b = std::min(b, n);
    //  A = [A11, A12]
    //      [*  , A22]
    int code = potrf_upper_sequential(curr_b, A, lda);
    if (code != 0) {
        std::cout << "Matrix indefinite. Returning early from potrf_upper.";
        return code;
    }
    //  A = [R11, A12]
    //      [*  , A22]
    if (curr_b < n) {
        T* R11 = A;                // shape (curr_b,     curr_b    )
        T* A12 = R11 + curr_b*lda; // shape (curr_b,     n - curr_b)
        T* A22 = A12 + curr_b;     // shape (n - curr_b, n - curr_b)
        blas::trsm(
            layout, blas::Side::Left, uplo, blas::Op::Trans, blas::Diag::NonUnit,
            curr_b, n - curr_b, (T) 1.0, R11, lda, A12, lda
        );
        blas::syrk(layout, uplo, blas::Op::Trans, 
            n - curr_b, curr_b, (T) -1.0, A12, lda, (T) 1.0, A22, lda
        );
        potrf_upper(n-curr_b, A22, lda, b);
    }
    return 0;
}

// If twice=true, then R will also be used as workspace, and must have length at least 2*n*n.
template <typename T>
void chol_qr(int64_t m, int64_t n, T* A, T* R, int64_t chol_block_size = 32, bool twice = false) {
    int64_t lda = m;
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    std::fill(R, R + n*n, (T) 0.0);
    blas::syrk(layout, uplo, blas::Op::Trans, n, m, (T) 1.0, A, lda, (T) 0.0, R, n);
    potrf_upper(n, R, n, chol_block_size);
    blas::trsm(layout, blas::Side::Right, uplo, blas::Op::NoTrans, blas::Diag::NonUnit, m, n, (T) 1.0, R, n, A, lda);
    if (twice) {
        T* R2 = R + n*n;
        chol_qr(m, n, A, R2, chol_block_size, false);
        RandBLAS::overwrite_triangle(layout, blas::Uplo::Lower, n, 1, R,  n);
        // now overwrite R = R2 R with TRMM (saying R2 is the triangular matrix)
        blas::trmm(layout, blas::Side::Left, uplo, blas::Op::NoTrans, blas::Diag::NonUnit, n, n, (T) 1.0, R2, n, R, n);
    }
    return;
}

// work must be length at least max(n, 2*b) * b
template <typename T>
void qr_block_cgs(int64_t m, int64_t n, T* A, T* R, int64_t ldr, T* work, int64_t b) {
    if (n > m)
        throw std::runtime_error("Invalid dimensions.");
    randblas_require(ldr >= n);

    b = std::min(b, n);
    auto layout = blas::Layout::ColMajor;
    using blas::Op;
    chol_qr(m, b, A, work, b, true);
    T one  = (T) 1.0;
    T zero = (T) 0.0;
    T* R1 = work;
    for (int64_t j = 0; j < b; ++j)
        blas::copy(b, R1 + b*j, 1, R + ldr*j, 1);

    if (b < n) {
        int64_t n_trail = n - b;
        T* A1 = A;           // A[:,  :b]
        T* A2 = A + m * b;   // A[:,  b:]
        T* R2 = R + ldr * b; // R[:b, b:]
        // Compute A1tA2 := A1' * A2 and then update A2 -= A1 *  A1tA2
        T* A1tA2 = work;
        blas::gemm(layout, Op::Trans,   Op::NoTrans, b, n_trail, m,  one, A1, m, A2,    m, zero, A1tA2, b);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n_trail, b, -one, A1, m, A1tA2, b,  one, A2,    m);
        // Copy A1tA2 to the appropriate place in R
        for (int64_t j = 0; j < n_trail; ++j) {
            blas::copy(b, A1tA2 + j*b, 1, R2 + j*ldr, 1);
        }
        qr_block_cgs(m, n_trail, A2, R2 + b, ldr, work, b);
    }
    return;
}

// We'll resize bigwork to be length at least (n*n + max(n, 2*b) * b).
template <typename T>
void qr_block_cgs2(int64_t m, int64_t n, T* A, T* R, std::vector<T> &bigwork, int64_t b = 64) {
    b = std::min(b, n);
    int64_t littlework_size = std::max(n, 2*b) * b;
    int64_t bigwork_size = n*n + littlework_size;
    if ((int64_t)bigwork.size() < bigwork_size) {
        bigwork.resize(bigwork_size);
    }
    T* R2 = bigwork.data();
    T* littlework = R2 + n*n;
    std::fill(R, R + n * n, (T) 0.0);
    qr_block_cgs(m, n, A, R, n, littlework, b);
    RandBLAS::overwrite_triangle(blas::Layout::ColMajor, blas::Uplo::Lower, n, 1, R, n);
    qr_block_cgs(m, n, A, R2, n, littlework, b);
    blas::trmm(
        blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
        n, n, 1.0, R2, n, R, n
    );
    return;
}

template <typename T>
bool extremal_eigvals_converged_gershgorin(int64_t n, T* G, T tol) {
    int64_t i_lb = 0;
    int64_t i_ub = 0;
    T upper = -std::numeric_limits<T>::infinity();
    T lower =  std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        T radius = 0.0;
        T center = G[i + i*n];
        for (int64_t j = 0; j < n; ++j) {
            if (i != j) {
                radius += std::abs(G[i * n + j]);
            }  
        }
        if (center + radius >= upper) {
            i_ub = i;
            upper = center + radius;
        }
        if (center - radius <= lower) {
            i_lb = i;
            lower = center - radius;
        }
    }
    T lower_center = G[i_lb + i_lb*n];
    T upper_center = G[i_ub + i_ub*n];
    T lower_radius = std::abs(lower - lower_center);
    T upper_radius = std::abs(upper - upper_center);
    bool lower_converged = lower_radius <= tol * lower_center;
    bool upper_converged = upper_radius <= tol * upper_center;
    bool converged = lower_converged && upper_converged;
    return converged;
}

/**
 * Use Cholesky iteration to compute all eigenvalues of a positive definite matrix A.
 * Run for at most "max_iters" iteration.
 * 
 * Use the Gershgorin circle theorem as a stopping criteria.
 *
 */
template <typename T>
int64_t posdef_eig_chol_iteration(int64_t n, T* A, T* eigvals, T reltol, int64_t max_iters, int64_t b = 8) {
    b = std::min(b, n);
    std::vector<T> work(n*n);
    T* G = work.data();
    using blas::Op;
    using blas::Layout;
    using blas::Uplo;
    int64_t iter = 0;
    bool converged = false;
    RandBLAS::RNGState state(1234567);
    std::vector<int64_t> pivots(n, 0);
    for (; iter < max_iters; ++iter) {
        potrf_upper(n, A, n, b);
        RandBLAS::overwrite_triangle(Layout::ColMajor, Uplo::Lower, n, 1, A, n);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, n, n, (T) 1.0, A, n, (T) 0.0, G, n);
        RandBLAS::symmetrize(Layout::ColMajor, Uplo::Upper, n, G, n);
        for (int64_t i = 0; i < n; ++i)
            eigvals[i] = G[i * n + i];
        converged = extremal_eigvals_converged_gershgorin(n, G, reltol);
        if (converged)
            break;
        blas::copy(n * n, G, 1, A, 1);
    }
    return (converged) ? iter : -iter;
}

template <typename T>
inline int64_t required_powermethod_iters(int64_t n, T p_fail, T tol) {
    T pi = 4*std::atan(1.0);
    int64_t expectation_bound = (int64_t) std::ceil(( 1.0 + std::log(std::sqrt(pi * (T)n)) )/ tol );

    T temp0 = 1 - tol;
    T temp1 = std::log(1 / temp0);
    T temp2 = tol * p_fail * p_fail;
    int64_t probability_bound_1 = (int64_t) std::log(std::exp(1.) +  (T)0.27 * temp0 * temp1 / temp2) / temp1;
    int64_t probability_bound_2 = (int64_t) std::log(std::sqrt(n) / p_fail) / temp1;
    int64_t probability_bound   = std::min(probability_bound_1, probability_bound_2);

    // std::cout << "(n, p, eps) = " << n << ", " << p_fail << ", " << tol << std::endl;
    // std::cout << "Power iters bound for expectation : " << expectation_bound << std::endl;
    // std::cout << "Power iters bound for probability : " << probability_bound << std::endl;
    int64_t num_iters = std::max(expectation_bound, probability_bound);
    return num_iters;
}

template <typename T, typename FUNC, typename RNG>
std::pair<T, RNGState<RNG>> power_method(int64_t n, FUNC &A, T* v, T tol, T failure_prob, const RNGState<RNG> &state) {
    auto next_state = RandBLAS::fill_dense_unpacked(blas::Layout::ColMajor, {n, 1}, n, 1, 0, 0, v, state);
    std::vector<T> work(n, 0.0);
    T* u = work.data();
    T norm = blas::nrm2(n, v, 1);
    blas::scal(n, (T)1.0/norm, v, 1);
    T lambda = 0.0;
    //
    int64_t num_iters = required_powermethod_iters(n, failure_prob, tol);
    for (int64_t iter = 0; iter < num_iters; ++iter) {
        A(v, u);
        lambda = blas::dot(n, v, 1, u, 1);
        blas::copy(n, u, 1, v, 1);
        norm = blas::nrm2(n, v, 1);
        blas::scal(n, (T)1.0/norm, v, 1);
    }
    return {lambda, next_state};
}

// Note that if we're only interested in subspace embedding distortion then it would suffice to just bound
// the eigenvalue of A-I with largest absolute value (which might be negative). If we went with that approach
// then we could make do with one run of a power method instead of running the power method on A and inv(A).
//
// The convergence results I know for the power method that don't require a spectral gap are specifically
// for PSD matrices. Now, we could just run the power method implicitly on the PSD matrix (A - I)^2.
// This require the same number of matrix-vector multiplications, but it remove the need for ever
// accessing inv(A) as a linear operator (which we do right now by decomposing A and forming invA explicitly,
// so we can get away with GEMV). That's useful if A is a fast operator (whether or not that's the case 
// might be delicate since it's a Gram matrix of a sketch S*U).
//
template <typename T, typename RNG>
std::tuple<T, T, RNGState<RNG>> exeigs_powermethod(int64_t n, const T* A, T* eigvecs, T tol, T failure_prob, const RNGState<RNG> &state,  std::vector<T> work) {
    auto layout = blas::Layout::ColMajor;
    RandBLAS::util::require_symmetric(layout, A, n, n, (T) 0.0);

    // Compute the dominant eigenpair. Nothing fancy here.
    auto A_func = [layout, A, n](const T* x, T* y) {
        blas::gemv(layout, blas::Op::NoTrans, n, n, (T) 1.0, A, n, x, 1, (T) 0.0, y, 1);
    };
    auto [lambda_max, next_state] = power_method<T>(n, A_func, eigvecs, tol, failure_prob, state);

    // To compute the smallest eigenpair we'll explicitly invert A. This requires
    // 2n^2 workspace: n^2 workspace for Cholesky of A (since we don't want to destroy A)
    // and another n^2 workspace for TRSMs with the Cholesky factor to get invA.
    //
    // Note: we *could* use less workspace if we were willing to access invA as a linear
    // operator using two calls to TRSV when needed. But this ends up being much slower
    // than explicit inversion for the values of (n, tol) that we care about.
    //
    if ((int64_t) work.size() < 2*n*n)
        work.resize(2*n*n);
    T* chol = work.data();
    blas::copy(n*n, A, 1, chol, 1);
    potrf_upper(n, chol, n);
    T* invA = chol + n*n;
    std::fill(invA, invA + n*n, 0.0);
    for (int i = 0; i < n; ++i)
        invA[i + i*n] = 1.0;
    auto uplo   = blas::Uplo::Upper;
    auto diag   = blas::Diag::NonUnit;
    blas::trsm(layout, blas::Side::Left, uplo, blas::Op::Trans,   diag, n, n, (T) 1.0, chol, n, invA, n);
    blas::trsm(layout, blas::Side::Left, uplo, blas::Op::NoTrans, diag, n, n, (T) 1.0, chol, n, invA, n);

    // Now that we have invA explicitly, getting its dominant eigenpair is effortless.
    auto invA_func = [layout, invA, n](const T* x, T* y) {
        blas::gemv(layout, blas::Op::NoTrans, n, n, (T) 1.0, invA, n, x, 1, (T) 0.0, y, 1);
        return;
    };
    auto [lambda_min, final_state] = power_method<T>(n, invA_func, eigvecs + n, tol, failure_prob, next_state);
    lambda_min = 1.0/lambda_min;

    return {lambda_max, lambda_min, final_state};
}

}
