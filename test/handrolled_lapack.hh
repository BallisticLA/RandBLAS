
#pragma once

#include "RandBLAS/util.hh"

#include <blas.hh>
#include <iostream>
#include <vector>
#include <cmath>

namespace hr_lapack {

template <typename T>
void potrf_upper_sequential(int64_t n, T* A, int64_t lda) {
    // Cache access is much better if the matrix is lower triangular.
    // Could implement as lower triangular and then call transpose_square.
    for (int64_t j = 0; j < n; ++j) {
        if (A[j + j * lda] <= 0) {
            std::stringstream s;
            s << "Cholesky failed at index " << j << " of " << n << ".";
            throw std::runtime_error(s.str());
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
}

template <typename T>
void potrf_upper(int64_t n, T* A, int64_t lda, int64_t b = 64) {
    randblas_require(b > 0);
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    int64_t curr_b = std::min(b, n);
    //  A = [A11, A12]
    //      [*  , A22]
    potrf_upper_sequential(curr_b, A, lda);
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
    return;
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
 * Use QR iteration to compute all eigenvalues of a positive definite matrix A.
 * Run for at most "max_iters" iteration.
 * 
 * In each iteration we'll perform some non-standard work to estimate the accuracy of
 * the eigenvalues. To explain, suppose "R" is our current triangular factor. We'll 
 * compute the Gram matrix G = R'R. The eigenvalues of G are the squares of the
 * singular values of R, and the singular values of R are equal to the eigenvaleus of A.
 * Therefore our estimate for the eigenvalues of A will be the square root of diag(G). 
 * We can terminate the algorithm once the relative radii of G's Gershgorin discs
 * fall below reltol^2.
 *
 * 
 */
template <typename T>
int64_t posdef_eig_qr_iteration(int64_t n, T* A, T* eigvals, T reltol, int64_t max_iters, int64_t b = 8) {
    b = std::min(b, n);

    int64_t subroutine_work_size = n*n + std::max(n, 2*b) * b;
    std::vector<T> work(n*n + subroutine_work_size);
    T* subroutine_work = work.data();
    T* G = subroutine_work;
    T* R = subroutine_work + subroutine_work_size;

    using blas::Op;
    using blas::Layout;
    using blas::Uplo;

    T sq_reltol = reltol * reltol;
    int64_t iter = 0;
    bool converged = false;
    for (; iter < max_iters; ++iter) {
        qr_block_cgs2(n, n, A, R, work, b);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, (T) 1.0, R, n, (T) 0.0, G, n);
        RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, n, G, n);
        for (int64_t i = 0; i < n; ++i)
            eigvals[i] = std::sqrt(G[i * n + i]);
        converged = extremal_eigvals_converged_gershgorin(n, G, sq_reltol);
        if (converged)
            break;
        // Update A = R * Q
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n, (T) 1.0, R, n, A, n, (T) 0.0, G, n);
        blas::copy(n * n, G, 1, A, 1);
    }
    return (converged) ? iter : -iter;
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

}