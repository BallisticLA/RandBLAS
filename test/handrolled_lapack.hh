
#pragma once

#include <blas.hh>
#include <iostream>
#include <vector>
#include <cmath>



// Function to perform Cholesky decomposition on a block
template <typename T>
void potrf_upper_colmajor_sequential(T* A, int64_t n, int64_t lda) {
    for (int64_t j = 0; j < n; ++j) {
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

// Function to perform blocked Cholesky decomposition
template <typename T>
void potrf_upper_colmajor(T* A, int64_t n, int64_t block_size = 64) {
    auto layout = blas::Layout::ColMajor;
    auto uplo = blas::Uplo::Upper;
    for (int64_t k = 0; k < n; k += block_size) {
        int64_t current_block_size = std::min(block_size, n - k);

        int64_t offset = k + k * n;
        // Perform Cholesky decomposition on the current block
        potrf_upper_colmajor_sequential(A + offset, current_block_size, n);

        if (k + current_block_size < n) {
            // Update the trailing submatrix
            blas::trsm(
                layout, blas::Side::Right, uplo, blas::Op::Trans, blas::Diag::NonUnit,
                n - k - current_block_size, current_block_size, 1.0,
                A + offset, n, A + k + current_block_size * n, n
            );
            blas::syrk(
                layout, uplo, blas::Op::NoTrans, 
                n - k - current_block_size, current_block_size, -1.0,
                A + k + current_block_size * n, n, 1.0,
                A + (k + current_block_size) + (k + current_block_size) * n, n
            );
        }
    }
}
