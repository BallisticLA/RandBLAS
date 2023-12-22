#ifndef randblas_sparse_data_coo_multiply
#define randblas_sparse_data_coo_multiply
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include <vector>
#include <algorithm>

namespace RandBLAS::sparse_data::coo {

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
static int64_t set_filtered_coo(
    // COO-format matrix data
    const T       *vals,
    const sint_t *rowidxs,
    const sint_t *colidxs,
    int64_t nnz,
    // submatrix bounds
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end,
    // COO-format submatrix data
    T       *new_vals,
    sint_t *new_rowidxs,
    sint_t *new_colidxs
) {
    int64_t new_nnz = 0;
    for (int64_t ell = 0; ell < nnz; ++ell) {
        if (
            row_start <= rowidxs[ell] && rowidxs[ell] < row_end &&
            col_start <= colidxs[ell] && colidxs[ell] < col_end
        ) {
            new_vals[new_nnz] = vals[ell];
            new_rowidxs[new_nnz] = rowidxs[ell] - row_start;
            new_colidxs[new_nnz] = colidxs[ell] - col_start;
            new_nnz += 1;
        }
    }
    return new_nnz;
}

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
static void apply_csc_to_vector_from_left(
    // CSC-format data
    const T *vals,
    sint_t *rowidxs,
    sint_t *colptr,
    // input-output vector data
    int64_t len_v,
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    int64_t i = 0;
    for (int64_t c = 0; c < len_v; ++c) {
        T scale = v[c * incv];
        while (i < colptr[c+1]) {
            int64_t row = rowidxs[i];
            Av[row * incAv] += (vals[i] * scale);
            i += 1;
        }
    }
}

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
static void apply_regular_csc_to_vector_from_left(
    // data for "regular CSC": CSC with fixed nnz per col,
    // which obviates the requirement for colptr.
    const T *vals,
    sint_t *rowidxs,
    int64_t col_nnz,
    // input-output vector data
    int64_t len_v,
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    for (int64_t c = 0; c < len_v; ++c) {
        T scale = v[c * incv];
        for (int64_t j = c * col_nnz; j < (c + 1) * col_nnz; ++j) {
            int64_t row = rowidxs[j];
            Av[row * incAv] += (vals[j] * scale);
        }
    }
}



template <typename T, RandBLAS::SignedInteger sint_t>
static void apply_coo_left(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    COOMatrix<T, sint_t> &A0,
    int64_t row_offset,
    int64_t col_offset,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A0.index_base == IndexBase::Zero);

    // Step 1: reduce to the case of CSC sort order.
    if (A0.sort != NonzeroSort::CSC) {
        auto orig_sort = A0.sort;
        sort_coo_data(NonzeroSort::CSC, A0);
        apply_coo_left(alpha, layout_B, layout_C, d, n, m, A0, row_offset, col_offset, B, ldb, C, ldc);
        sort_coo_data(orig_sort, A0);
        return;
    }

    // Step 2: make a CSC-sort-order COOMatrix that represents the desired submatrix of A.
    //      While we're at it, reduce to the case when alpha = 1.0 by scaling the values
    //      of the matrix we just created.
    int64_t A_nnz;
    int64_t A0_nnz = A0.nnz;
    std::vector<sint_t> A_rows(A0_nnz, 0);
    std::vector<sint_t> A_colptr(std::max(A0_nnz, m + 1), 0);
    std::vector<T> A_vals(A0_nnz, 0.0);
    A_nnz = set_filtered_coo(
        A0.vals, A0.rows, A0.cols, A0.nnz,
        col_offset, col_offset + m,
        row_offset, row_offset + d,
        A_vals.data(), A_rows.data(), A_colptr.data()
    );
    blas::scal<T>(A_nnz, alpha, A_vals.data(), 1);
    sorted_nonzero_locations_to_pointer_array(A_nnz, A_colptr.data(), m);
    bool fixed_nnz_per_col = true;
    for (int64_t ell = 2; (ell < m + 1) && fixed_nnz_per_col; ++ell)
        fixed_nnz_per_col = (A_colptr[1] == A_colptr[ell]);

    // Step 3: Apply "A" to the left of B to get C += A*B.
    //      3.1: set stride information (we can't use structured bindings because of an OpenMP bug)
    //      3.2: iterate over the columns of the matrix B.
    //      3.3: compute the matrix-vector products
    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    s = layout_to_strides(layout_C, ldc);
    auto C_inter_col_stride = s.inter_col_stride;
    auto C_inter_row_stride = s.inter_row_stride;

    #pragma omp parallel default(shared)
    {
        const T *B_col = nullptr;
        T *C_col = nullptr;
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < n; k++) {
            B_col = &B[B_inter_col_stride * k];
            C_col = &C[C_inter_col_stride * k];
            if (fixed_nnz_per_col) {
                apply_regular_csc_to_vector_from_left<T>(
                    A_vals.data(), A_rows.data(), A_colptr[1],
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                );
            } else {
                apply_csc_to_vector_from_left<T>(
                    A_vals.data(), A_rows.data(), A_colptr.data(),
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                ); 
            }
        }
    }
    return;
}

template <typename T>
void lspgemm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t d, // C is d-by-n
    int64_t n, // \op(B) is m-by-n
    int64_t m, // \op(A) is d-by-m
    T alpha,
    COOMatrix<T> &A,
    int64_t row_offset,
    int64_t col_offset,
    const T *B,
    int64_t ldb,
    T beta,
    T *C,
    int64_t ldc
) {
    // handle applying a transposed sparse sketching operator.
    if (opA == blas::Op::Trans) {
        auto At = transpose(A);
        lspgemm(
            layout, blas::Op::NoTrans, opB,
            d, n, m, alpha, At, col_offset, row_offset,
            B, ldb, beta, C, ldc
        );
        return; 
    }
    // Below this point, we can assume A is not transposed.
    randblas_require(A.n_rows >= d);
    randblas_require(A.n_cols >= m);

    // Dimensions of B, rather than \op(B)
    blas::Layout layout_C = layout;
    blas::Layout layout_opB;
    int64_t rows_B, cols_B;
    if (opB == blas::Op::NoTrans) {
        rows_B = m;
        cols_B = n;
        layout_opB = layout;
    } else {
        rows_B = n;
        cols_B = m;
        layout_opB = (layout == blas::Layout::ColMajor) ? blas::Layout::RowMajor : blas::Layout::ColMajor;
    }

    // Check dimensions and compute C = beta * C.
    //      Note: both B and C are checked based on "layout"; B is *not* checked on layout_opB.
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= rows_B);
        randblas_require(ldc >= d);
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(d, beta, &C[i*ldc], 1);
    } else {
        randblas_require(ldc >= n);
        randblas_require(ldb >= cols_B);
        for (int64_t i = 0; i < d; ++i)
            RandBLAS::util::safe_scal(n, beta, &C[i*ldc], 1);
    }

    // compute the matrix-matrix product
    if (alpha != 0)
        apply_coo_left(alpha, layout_opB, layout_C, d, n, m, A, row_offset, col_offset, B, ldb, C, ldc);
    return;
}

template <typename T>
void rspgemm(
    blas::Layout layout,
    blas::Op opB,
    blas::Op opA,
    int64_t m, // C is m-by-d
    int64_t d, // op(A) is n-by-d
    int64_t n, // op(B) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    COOMatrix<T> &B0,
    int64_t i_off,
    int64_t j_off,
    T beta,
    T *C,
    int64_t ldc
) { 
    //
    // Compute C = op(B) op(submat(A)) by reduction to LSPGEMM. We start with
    //
    //      C^T = op(submat(A))^T op(B)^T.
    //
    // Then we interchange the operator "op(*)" in op(B) and (*)^T:
    //
    //      C^T = op(submat(A))^T op(B^T).
    //
    // We tell LSPGEMM to process (C^T) and (B^T) in the opposite memory layout
    // compared to the layout for (B, C).
    // 
    using blas::Layout;
    using blas::Op;
    auto trans_opA = (opA == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    lspgemm(
        trans_layout, trans_opA, opB,
        d, m, n, alpha, B0, i_off, j_off, A, lda, beta, C, ldc
    );
}


} // end namespace

#endif
