
#pragma once

#include "RandBLAS/util.hh"
#include "RandBLAS/dense_skops.hh"

#include <blas.hh>
#include <iostream>
#include <vector>
#include <cmath>

namespace hr_lapack {

template <typename T>
int potrf_upper_sequential(int64_t n, T* A, int64_t lda) {
    // Cache access is much better if the matrix is lower triangular.
    // Could implement as lower triangular and then call transpose_square.
    for (int64_t j = 0; j < n; ++j) {
        if (A[j + j * lda] <= 0) {
            // std::stringstream s;
            std::cout << "Cholesky failed at index " << j << " of " << n << ".";
            // throw std::runtime_error(s.str());
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
        RandBLAS::util::overwrite_triangle(layout, blas::Uplo::Lower, n, 1, (T) 0.0, R,  n);
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
    RandBLAS::util::overwrite_triangle(blas::Layout::ColMajor, blas::Uplo::Lower, n, 1, (T) 0.0, R, n);
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
    // int64_t n, const T* A, int64_t &k, int64_t* S,  T* F, int64_t b, STATE state
    int64_t iter = 0;
    bool converged = false;
    RandBLAS::RNGState state(1234567);
    std::vector<int64_t> pivots(n, 0);
    for (; iter < max_iters; ++iter) {
        potrf_upper(n, A, n, b);
        RandBLAS::util::overwrite_triangle(Layout::ColMajor, Uplo::Lower, n, 1, (T) 0.0, A, n);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, n, n, (T) 1.0, A, n, (T) 0.0, G, n);
        RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, n, G, n);
        for (int64_t i = 0; i < n; ++i)
            eigvals[i] = G[i * n + i];
        converged = extremal_eigvals_converged_gershgorin(n, G, reltol);
        if (converged)
            break;
        blas::copy(n * n, G, 1, A, 1);
    }
    return (converged) ? iter : -iter;
}

template <typename T, typename FUNC, typename RNG>
T power_method(int64_t n, FUNC &A, T* v, T tol, RandBLAS::RNGState<RNG> state) {
    RandBLAS::fill_dense(blas::Layout::ColMajor, {n, 1}, n, 1, 0, 0, v, state);
    std::vector<T> work(n, 0.0);
    T* u = work.data();
    T norm = blas::nrm2(n, v, 1);
    blas::scal(n, (T)1.0/norm, v, 1);
    T lambda = 0.0;
    T pi = 4*std::atan(1.0);
    int64_t max_iter = (int64_t) std::ceil(( 1.0 + std::log(std::sqrt(pi * (T)n)) )/ tol );
    for (int64_t iter = 0; iter < max_iter; ++iter) {
        A(v, u);
        lambda = blas::dot(n, v, 1, u, 1);
        blas::copy(n, u, 1, v, 1);
        norm = blas::nrm2(n, v, 1);
        blas::scal(n, (T)1.0/norm, v, 1);
    }
    return lambda;
}

/**
 * @brief Lanczos Algorithm to find the largest eigenvalue of a symmetric positive definite matrix.
 *
 * @tparam scalar_t a real scalar type.
 * @param A a callable, where an evaluation of the form A(x, y) overwrites y = A*x.
 * @param n The order of the matrix (number of rows and columns).
 * @param m The number of Lanczos iterations to perform.
 * @param eigenvector Pointer to the output vector that will store the approximated eigenvector corresponding to the largest eigenvalue.
 * @param max_iter The maximum number of iterations for the power method used to compute the largest eigenvalue of the tridiagonal matrix (default is 1000).
 * @param tol The tolerance for convergence of the power method (default is 1e-6).
 * @return The approximated largest eigenvalue of the matrix.
 * 
 * @endcode
 */
template <typename real_t, typename FUNC>
std::pair<real_t,int64_t> lanczos(int64_t n, FUNC &A, int64_t lan_iter, int64_t pow_iter, real_t tol = 1e-6) {
    // Allocate workspace
    std::vector<real_t> workspace(4 * n + 4 * lan_iter + 1 + n*(lan_iter+1), 0.0);
    real_t* u     = workspace.data();
    real_t* v     = u + n;
    real_t* w     = v + n;
    real_t* alpha = w + n;
    real_t* beta  = alpha + lan_iter;
    real_t* b0    = beta  + lan_iter + 1;
    real_t* b1    = b0    + lan_iter;
    real_t* Q     = b1    + lan_iter;

    auto normalize = [](int64_t k, real_t* vec) {
        real_t scale = ((real_t) 1.0) / blas::nrm2(k, vec, 1);
        blas::scal(k, scale, vec, 1);
    };

    RandBLAS::RNGState state(8739);
    auto next_state = RandBLAS::fill_dense(blas::Layout::ColMajor, {n, 1}, n, 1, 0, 0, w, state);
    normalize(n, w);
    blas::copy(n, w, 1, Q, 1);
    A(w, v); // v =  Aw
    alpha[0] = blas::dot(n, w, 1, v, 1); // alpha[0] = w'v
    for (int i = 0; i < n; ++i) { v[i] -= alpha[0]*w[i]; } // v = v - alpha[0] w
    beta[0] = blas::nrm2(n, v, 1); // beta[0] = ||v||
    int64_t k = 0;
    while (k < lan_iter && beta[k] >= tol) {
        for (int i = 0; i < n; ++i) { real_t t = w[i];  w[i] = v[i] / beta[k]; v[i] = -beta[k] * t; }
        A(w, u);
        for (int i = 0; i < n; ++i) { v[i] += u[i]; }
        k = k + 1;
        alpha[k] = blas::dot(n, w, 1, v, 1);
        for (int i = 0; i < n; ++i) { v[i] -= alpha[k]*w[i]; }
        //
        // Update the Lanczos vectors; using complete reorthogonalization with modified Gram-Schmidt
        //
        real_t* q_k = Q + (k-1)*n;
        blas::copy(n, w, 1, q_k, 1);
        real_t norm_w = blas::nrm2(n, q_k, 1);
        blas::scal(n, (real_t)1.0 / norm_w, q_k, 1);
        // update q_k = (I - Q_k Q_k') q_k, where Q_k = Q[:, 1:(k-1)]
        //      u    = Q_k' q_k
        //      q_k -= Q_k u
        blas::gemv(blas::Layout::ColMajor,   blas::Op::Trans, n, k-1, (real_t)  1.0, Q, n, q_k, 1, (real_t) 0.0,   u, 1);
        blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, n, k-1, (real_t) -1.0, Q, n,   u, 1, (real_t) 1.0, q_k, 1);
        // update v = (I - Q_{k+1} Q_{k+1}') v, which should do nothing in exact arithmetic.
        blas::gemv(blas::Layout::ColMajor,   blas::Op::Trans, n, k,  1.0, Q, n, v, 1, (real_t) 0.0, u, 1);
        blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, n, k, -1.0, Q, n, u, 1, (real_t) 1.0, v, 1);
        beta[k] = blas::nrm2(n, v, 1);
    }
    lan_iter = k;

    auto T_func = [lan_iter, alpha, beta](const real_t* x, real_t* y) {
        // Apply the tridiagonal matrix defined by (alpha, beta) to
        // the vector x and store the result in y.
        std::fill(y, y + lan_iter, 0.0);
        for (int64_t i = 0; i < lan_iter; ++i) {
            y[i] += alpha[i] * x[i];
            if (i > 0) {
                y[i] += beta[i - 1] * x[i - 1];
            }
            if (i < lan_iter - 1) {
                y[i] += beta[i] * x[i + 1];
            }
        }
        return;
    };

    // Compute the largest lambda of the tridiagonal matrix T
    // For simplicity, we use the power method on T
    RandBLAS::fill_dense(blas::Layout::ColMajor, {lan_iter, 1}, lan_iter, 1, 0, 0, b0, next_state);
    normalize(lan_iter, b0);
    real_t lambda = 0.0;
    int64_t iter = 0;
    for (; iter < pow_iter; ++iter) {
        T_func(b0, b1);
        real_t lambda_next = blas::dot(lan_iter, b0, 1, b1, 1);
        if (std::abs(lambda_next - lambda) < tol) {
            lambda = lambda_next;
            break;
        }
        lambda = lambda_next;
        blas::copy(lan_iter, b1, 1, b0, 1);
        normalize(lan_iter, b0);
    }
    return {lambda, iter};
}

}
