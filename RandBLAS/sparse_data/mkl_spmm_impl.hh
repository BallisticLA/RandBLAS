// Copyright, 2026. See LICENSE for copyright holder information.
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

#pragma once

#include "RandBLAS/config.h"

#if defined(RandBLAS_HAS_MKL)

// Ensure MKL uses 64-bit integers when BLAS++ was built with ILP64.
#if defined(BLAS_ILP64) && !defined(MKL_ILP64)
#define MKL_ILP64
#endif

#include <mkl_spblas.h>
#include <type_traits>
#include <stdexcept>

#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"

namespace RandBLAS::sparse_data::mkl {

// ============================================================================
// RAII wrapper for MKL sparse_matrix_t handles.
// Automatically calls mkl_sparse_destroy on scope exit.
// ============================================================================
struct MKLSparseHandle {
    sparse_matrix_t handle = nullptr;

    MKLSparseHandle() = default;
    ~MKLSparseHandle() {
        if (handle)
            mkl_sparse_destroy(handle);
    }

    // Non-copyable
    MKLSparseHandle(const MKLSparseHandle&) = delete;
    MKLSparseHandle& operator=(const MKLSparseHandle&) = delete;

    // Movable
    MKLSparseHandle(MKLSparseHandle&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }
    MKLSparseHandle& operator=(MKLSparseHandle&& other) noexcept {
        if (this != &other) {
            if (handle) mkl_sparse_destroy(handle);
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }
};

// ============================================================================
// Helper: check MKL status and throw on error
// ============================================================================
inline void check_mkl_status(sparse_status_t status, const char* func_name) {
    if (status != SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("MKL sparse BLAS error in ") + func_name +
            ": status code " + std::to_string(static_cast<int>(status))
        );
    }
}

// ============================================================================
// Helper: convert blas::Layout to MKL sparse_layout_t
// ============================================================================
inline sparse_layout_t to_mkl_layout(blas::Layout layout) {
    return (layout == blas::Layout::ColMajor)
        ? SPARSE_LAYOUT_COLUMN_MAJOR
        : SPARSE_LAYOUT_ROW_MAJOR;
}

// ============================================================================
// Helper: convert blas::Op to MKL sparse_operation_t
// ============================================================================
inline sparse_operation_t to_mkl_op(blas::Op op) {
    switch (op) {
        case blas::Op::NoTrans: return SPARSE_OPERATION_NON_TRANSPOSE;
        case blas::Op::Trans:   return SPARSE_OPERATION_TRANSPOSE;
        default:                return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }
}

// ============================================================================
// Type-dispatched MKL sparse handle creation from CSR
// Zero-copy: just wraps existing arrays.
// ============================================================================
template <typename T, typename sint_t>
MKLSparseHandle make_mkl_handle_csr(const CSRMatrix<T, sint_t>& A) {
    static_assert(sizeof(sint_t) == sizeof(MKL_INT),
        "RandBLAS MKL backend: sparse matrix index type must match MKL_INT size. "
        "With ILP64, both should be 64-bit.");

    MKLSparseHandle h;
    // MKL CSR uses 4-array format: rows_start, rows_end, col_indx, values.
    // Standard 3-array CSR (rowptr) maps as: rows_start = rowptr, rows_end = rowptr+1.
    auto* rowptr = reinterpret_cast<MKL_INT*>(A.rowptr);
    auto* colidxs = reinterpret_cast<MKL_INT*>(A.colidxs);

    sparse_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_csr(
            &h.handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)A.n_rows, (MKL_INT)A.n_cols,
            rowptr, rowptr + 1, colidxs, A.vals
        );
    } else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_create_csr(
            &h.handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)A.n_rows, (MKL_INT)A.n_cols,
            rowptr, rowptr + 1, colidxs, A.vals
        );
    } else {
        static_assert(sizeof(T) == 0, "MKL sparse BLAS only supports float and double.");
    }
    check_mkl_status(status, "mkl_sparse_create_csr");
    return h;
}

// ============================================================================
// Type-dispatched MKL sparse handle creation from CSC
// Zero-copy: just wraps existing arrays.
// ============================================================================
template <typename T, typename sint_t>
MKLSparseHandle make_mkl_handle_csc(const CSCMatrix<T, sint_t>& A) {
    static_assert(sizeof(sint_t) == sizeof(MKL_INT),
        "RandBLAS MKL backend: sparse matrix index type must match MKL_INT size.");

    MKLSparseHandle h;
    auto* colptr = reinterpret_cast<MKL_INT*>(A.colptr);
    auto* rowidxs = reinterpret_cast<MKL_INT*>(A.rowidxs);

    sparse_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_csc(
            &h.handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)A.n_rows, (MKL_INT)A.n_cols,
            colptr, colptr + 1, rowidxs, A.vals
        );
    } else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_create_csc(
            &h.handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)A.n_rows, (MKL_INT)A.n_cols,
            colptr, colptr + 1, rowidxs, A.vals
        );
    } else {
        static_assert(sizeof(T) == 0, "MKL sparse BLAS only supports float and double.");
    }
    check_mkl_status(status, "mkl_sparse_create_csc");
    return h;
}

// ============================================================================
// MKL sparse handle creation from COO.
// Creates a COO handle, then converts to CSR internally via MKL.
// The returned handle owns the CSR data allocated by MKL.
// ============================================================================
template <typename T, typename sint_t>
MKLSparseHandle make_mkl_handle_coo(const COOMatrix<T, sint_t>& A) {
    static_assert(sizeof(sint_t) == sizeof(MKL_INT),
        "RandBLAS MKL backend: sparse matrix index type must match MKL_INT size.");

    MKLSparseHandle coo_h;
    auto* rows = reinterpret_cast<MKL_INT*>(A.rows);
    auto* cols = reinterpret_cast<MKL_INT*>(A.cols);

    sparse_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_coo(
            &coo_h.handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)A.n_rows, (MKL_INT)A.n_cols,
            (MKL_INT)A.nnz, rows, cols, A.vals
        );
    } else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_create_coo(
            &coo_h.handle, SPARSE_INDEX_BASE_ZERO,
            (MKL_INT)A.n_rows, (MKL_INT)A.n_cols,
            (MKL_INT)A.nnz, rows, cols, A.vals
        );
    } else {
        static_assert(sizeof(T) == 0, "MKL sparse BLAS only supports float and double.");
    }
    check_mkl_status(status, "mkl_sparse_create_coo");

    // Convert COO to CSR (MKL allocates and owns the CSR data).
    MKLSparseHandle csr_h;
    status = mkl_sparse_convert_csr(coo_h.handle, SPARSE_OPERATION_NON_TRANSPOSE, &csr_h.handle);
    check_mkl_status(status, "mkl_sparse_convert_csr");

    // coo_h is destroyed automatically (frees COO handle, not our arrays).
    // csr_h owns the internally allocated CSR arrays.
    return csr_h;
}

// ============================================================================
// Generic handle creation: dispatches based on sparse matrix format.
// ============================================================================
template <SparseMatrix SpMat>
MKLSparseHandle make_mkl_handle(const SpMat& A) {
    using T = typename SpMat::scalar_t;
    using sint_t = typename SpMat::index_t;

    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;

    if constexpr (is_csr) {
        return make_mkl_handle_csr(A);
    } else if constexpr (is_csc) {
        return make_mkl_handle_csc(A);
    } else if constexpr (is_coo) {
        return make_mkl_handle_coo(A);
    } else {
        static_assert(sizeof(SpMat) == 0, "Unsupported sparse matrix format for MKL backend.");
    }
}

// ============================================================================
// MKL-accelerated left_spmm: C = alpha * op(A) * op(B) + beta * C
//   where A is sparse, B and C are dense.
//
// NOTE: The caller (spmm_dispatch.hh) has already applied beta scaling to C,
//   so this function is called with beta=0.
//
// For COO matrices with submatrix offsets (ro_a, co_a != 0), we fall back
//   to the existing RandBLAS implementation since MKL doesn't support
//   submatrix views on COO.
// ============================================================================
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
bool mkl_left_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t d, int64_t n, int64_t m,
    T alpha,
    const SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    const T *B,
    int64_t ldb,
    T beta,
    T *C,
    int64_t ldc
) {
    using sint_t = typename SpMat::index_t;
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;

    // MKL's mkl_sparse_d_mm does not support CSC format (returns NOT_SUPPORTED).
    // Fall back to hand-rolled kernels for CSC matrices.
    if constexpr (is_csc)
        return false;

    // For COO matrices, RandBLAS allows A.n_rows >= d and A.n_cols >= m
    // (operating on a submatrix). MKL operates on the full matrix, so we
    // can only use MKL when dimensions match exactly and offsets are zero.
    if constexpr (is_coo) {
        if (ro_a != 0 || co_a != 0)
            return false;
        if (A.n_rows != d || A.n_cols != m)
            return false;
    }

    // MKL's mkl_sparse_d_mm uses a single layout parameter for both B and C.
    // When opB == Trans, the caller's B and C have different effective layouts,
    // which MKL can't express. Fall back.
    if (opB != blas::Op::NoTrans)
        return false;

    auto h = make_mkl_handle(A);
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_mm(
            to_mkl_op(opA), alpha, h.handle, descr,
            to_mkl_layout(layout),
            B, (MKL_INT)n, (MKL_INT)ldb,
            beta, C, (MKL_INT)ldc
        );
    } else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_mm(
            to_mkl_op(opA), alpha, h.handle, descr,
            to_mkl_layout(layout),
            B, (MKL_INT)n, (MKL_INT)ldb,
            beta, C, (MKL_INT)ldc
        );
    }
    check_mkl_status(status, "mkl_sparse_mm");
    return true;  // signal: MKL handled it
}

// ============================================================================
// MKL-accelerated sparse x sparse -> dense: C = op(A) * op(B)
//   where A and B are sparse, C is dense.
//
// Uses mkl_sparse_d_spmmd (or _s for float).
// ============================================================================
template <SparseMatrix SpMat1, SparseMatrix SpMat2,
          typename T = typename SpMat1::scalar_t>
void mkl_spgemm_to_dense(
    blas::Layout layout,
    blas::Op opA,
    int64_t m, int64_t n,
    T alpha,
    const SpMat1 &A,
    const SpMat2 &B,
    T beta,
    T *C,
    int64_t ldc
) {
    static_assert(std::is_same_v<T, typename SpMat2::scalar_t>,
        "Both sparse matrices must have the same scalar type.");

    auto hA = make_mkl_handle(A);
    auto hB = make_mkl_handle(B);

    // mkl_sparse_d_spmmd computes C = op(A) * B (no alpha/beta).
    // We handle alpha/beta manually:
    //   1. Scale C by beta
    //   2. Compute temp = op(A) * B into a temporary (or directly into C if alpha=1, beta=0)
    //   3. C = alpha * temp + C (if needed)

    if (alpha == (T)0) {
        // Just scale C by beta
        if (beta == (T)0) {
            int64_t total = (layout == blas::Layout::ColMajor) ? ldc * n : ldc * m;
            std::fill(C, C + total, (T)0);
        } else if (beta != (T)1) {
            // Scale each column/row of C
            if (layout == blas::Layout::ColMajor) {
                for (int64_t j = 0; j < n; ++j)
                    blas::scal(m, beta, &C[j * ldc], 1);
            } else {
                for (int64_t i = 0; i < m; ++i)
                    blas::scal(n, beta, &C[i * ldc], 1);
            }
        }
        return;
    }

    // Fast path: alpha=1, beta=0 → write directly into C
    bool direct_write = (alpha == (T)1 && beta == (T)0);

    T* target = C;
    std::vector<T> temp_buf;
    if (!direct_write) {
        int64_t buf_size = (layout == blas::Layout::ColMajor) ? ldc * n : ldc * m;
        temp_buf.resize(buf_size, (T)0);
        target = temp_buf.data();
    }

    sparse_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_spmmd(
            to_mkl_op(opA), hA.handle, hB.handle,
            to_mkl_layout(layout), target, (MKL_INT)ldc
        );
    } else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_spmmd(
            to_mkl_op(opA), hA.handle, hB.handle,
            to_mkl_layout(layout), target, (MKL_INT)ldc
        );
    }
    check_mkl_status(status, "mkl_sparse_spmmd");

    if (!direct_write) {
        // C = alpha * target + beta * C
        if (layout == blas::Layout::ColMajor) {
            for (int64_t j = 0; j < n; ++j) {
                blas::scal(m, beta, &C[j * ldc], 1);
                blas::axpy(m, alpha, &target[j * ldc], 1, &C[j * ldc], 1);
            }
        } else {
            for (int64_t i = 0; i < m; ++i) {
                blas::scal(n, beta, &C[i * ldc], 1);
                blas::axpy(n, alpha, &target[i * ldc], 1, &C[i * ldc], 1);
            }
        }
    }
}

} // namespace RandBLAS::sparse_data::mkl

#endif // RandBLAS_HAS_MKL
