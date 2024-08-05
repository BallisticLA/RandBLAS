
#pragma once

#include <blas.hh>
#include <iostream>
#include <vector>
#include <cmath>



template <typename T>
void potrf_upper_colmajor_sequential(int64_t n, T* A, int64_t lda) {
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
void potrf_upper_colmajor(int64_t n, T* A, int64_t b = 64) {
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    for (int64_t k = 0; k < n; k += b) {
        int64_t curr_b = std::min(b, n - k);

        int64_t offset = k + k * n;
        // Perform Cholesky decomposition on the current block
        potrf_upper_colmajor_sequential(curr_b, A + offset, n);

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
void chol_qr_colmajor(int64_t m, int64_t n, T* A, T* R, int64_t chol_block_size = 32) {
    int64_t lda = m;
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    std::fill(R, R + n*n, (T) 0.0);
    blas::syrk(layout, uplo, blas::Op::Trans, n, m, (T) 1.0, A, lda, (T) 0.0, R, n);
    potrf_upper_colmajor(n, R, chol_block_size);
    blas::trsm(layout, blas::Side::Right, uplo, blas::Op::NoTrans, blas::Diag::NonUnit, m, n, (T) 1.0, R, n, A, lda);
}

template <typename T>
void block_gram_schmidt(int64_t m, int64_t n, T* A, std::vector<T> &work, int64_t b = 64) {
    b = std::min(b, n);
    if (work.size() < n*b) {
        work.resize(n*b);
    }
    auto layout = blas::Layout::ColMajor;
    using blas::Op;
    chol_qr_colmajor(m, b, A, work.data(), b);
    T one  = 1.0;
    T zero = 0.0;
    if (b < n) {
        int64_t n_trail = n - b;
        T* A1 = A;         // A[:, :b]
        T* A2 = A + b * m; // A[:, b:]
        // Compute A1tA2 := A1' * A2 and then update A2 -= A1 *  A1tA2
        T* A1tA2 = work.data();
        blas::gemm(layout, Op::Trans,   Op::NoTrans, b, n_trail, m,  one, A1, m, A2,    m, zero, A1tA2, b);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n_trail, b, -one, A1, m, A1tA2, b,  one, A2,    m);
        block_gram_schmidt(m, n - b, A + b * m, work, b);
    }
}

template <typename T>
void block_chol_qr(int64_t n, T* A, T* R, int64_t b = 64) {
    b = std::min(b, n);
    std::vector<T> work_orth(n*b);
    std::vector<T> work_R(n*n);
    T* A_copy = work_R.data();
    blas::copy(n*n, A, 1, A_copy, 1);
    block_gram_schmidt(n, n, A, work_orth, b);
    using blas::Layout;
    using blas::Op;
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n, (T)1.0, A, n, A_copy, n, (T) 0.0, R, n);
}

// TODOs:
//  1) change block_gram_schmidt(...) to use modified Gram-Schmidt instead of classic Gram-Schmit.
//  2) merge block_gram_schmidt and block_chol_qr so that block_gram_schmidt builds R as it goes.
//      --> This might require some attention to submatrix pointers for R.
///
