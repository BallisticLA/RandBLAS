
#pragma once

#include "RandBLAS/util.hh"

#include <blas.hh>
#include <iostream>
#include <vector>
#include <cmath>


template <typename T>
void potrf_upper_sequential(int64_t n, T* A, int64_t lda) {
    for (int64_t j = 0; j < n; ++j) {
        if (A[j + j * lda] <= 0) {
            std::stringstream s;
            s << "Cholesky failed at index " << j << " of " << n ".\n";
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
void potrf_upper(int64_t n, T* A, int64_t b = 64) {
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    for (int64_t k = 0; k < n; k += b) {
        int64_t curr_b = std::min(b, n - k);

        int64_t offset = k + k * n;
        // Perform Cholesky decomposition on the current block
        potrf_upper_sequential(curr_b, A + offset, n);

        if (k + curr_b < n) {
            // Update the trailing submatrix
            blas::trsm(
                layout, blas::Side::Right, uplo, blas::Op::Trans, blas::Diag::NonUnit,
                n - k - curr_b, curr_b, 1.0,
                A + offset, n, A + k + curr_b * n, n
            );
            blas::syrk(
                layout, uplo, blas::Op::NoTrans, 
                n - k - curr_b, curr_b, -1.0,
                A + k + curr_b * n, n, 1.0,
                A + (k + curr_b) + (k + curr_b) * n, n
            );
        }
    }
}

template <typename T>
void chol_qr(int64_t m, int64_t n, T* A, T* R, int64_t chol_block_size = 32) {
    int64_t lda = m;
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    std::fill(R, R + n*n, (T) 0.0);
    blas::syrk(layout, uplo, blas::Op::Trans, n, m, (T) 1.0, A, lda, (T) 0.0, R, n);
    potrf_upper(n, R, chol_block_size);
    blas::trsm(layout, blas::Side::Right, uplo, blas::Op::NoTrans, blas::Diag::NonUnit, m, n, (T) 1.0, R, n, A, lda);
}

template <typename T>
void qr_block_cgs(int64_t m, int64_t n, T* A, T* R, std::vector<T>& work, int64_t b) {
    if (n > m)
        throw std::runtime_error("Invalid dimensions.");

    b = std::min(b, n);
    if (work.size() < n * b) {
        work.resize(n * b);
    }
    auto layout = blas::Layout::ColMajor;
    using blas::Op;
    chol_qr(m, b, A, work.data(), b);
    T one  = (T) 1.0;
    T zero = (T) 0.0;
    T* R1 = work.data();
    for (int64_t j = 0; j < b; ++j)
        blas::copy(b, R1 + b*j, 1, R + n*j, 1);

    if (b < n) {
        int64_t n_trail = n - b;
        T* A1 = A;         // A[:, :b]
        T* A2 = A + m * b; // A[:, b:]
        T* R2 = R + n * b; // R[:b, b:]
        // Compute A1tA2 := A1' * A2 and then update A2 -= A1 *  A1tA2
        T* A1tA2 = work.data();
        blas::gemm(layout, Op::Trans,   Op::NoTrans, b, n_trail, m,  one, A1, m, A2,    m, zero, A1tA2, b);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n_trail, b, -one, A1, m, A1tA2, b,  one, A2,    m);
        // Copy A1tA2 to the appropriate place in R
        for (int64_t j = 0; j < n_trail; ++j) {
            blas::copy(b, A1tA2 + j*b, 1, R2 + j*n, 1);
        }
        qr_block_cgs(m, n_trail, A + b * m, R + b * n + b, work, b);
    }
}


template <typename T>
void qr_block_cgs(int64_t n, T* A, T* R, int64_t b = 64) {
    b = std::min(b, n);
    std::vector<T> work(n * b);
    std::fill(R, R + n * n, (T) 0.0);
    qr_block_cgs(n, n, A, R, work, b);
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
 * We can terminate the algorithm once the relative radius of each Gershgorin disc
 * falls below reltol.
 *
 * 
 */
template <typename T>
void eig_qr_iteration(int64_t n, T* A, T* eigvals, T reltol, int64_t max_iters, int64_t b = 8) {
    std::vector<T> workspace(2 * n * n);
    T* R = workspace.data();
    T* G = R + n * n;

    using blas::Op;
    using blas::Layout;
    using blas::Uplo;

    for (int64_t iter = 0; iter < max_iters; ++iter) {
        qr_block_cgs(n, n, A, R, workspace, b);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, (T) 1.0, R, n, (T) 0.0, G, n);
        RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, G, n, n);
        for (int64_t i = 0; i < n; ++i)
            eigvals[i] = std::sqrt(G[i * n + i]);
        
        bool converged = true;
        int64_t i = 0;
        while(i < n && converged) {
            T radius = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                if (i != j)
                    radius += std::abs(G[i * n + j]);
            }
            converged = radius <= reltol*G[i * n + i];
            ++i;
        }
        if (converged)
            break;
        // Update A = R * Q
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n, (T) 1.0, R, n, A, n, (T) 0.0, G, n);
        blas::copy(n * n, G, 1, A, 1);
    }
    return;
}
