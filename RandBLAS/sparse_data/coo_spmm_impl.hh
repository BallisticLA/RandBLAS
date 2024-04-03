#ifndef randblas_sparse_data_coo_multiply
#define randblas_sparse_data_coo_multiply
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csc_spmm_impl.hh"
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



template <typename T, RandBLAS::SignedInteger sint_t>
static void apply_coo_left_jki_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    COOMatrix<T, sint_t> &A0,
    int64_t ro_a,
    int64_t co_a,
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
        apply_coo_left_jki_p11(alpha, layout_B, layout_C, d, n, m, A0, ro_a, co_a, B, ldb, C, ldc);
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
        co_a, co_a + m,
        ro_a, ro_a + d,
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
        for (int64_t j = 0; j < n; j++) {
            B_col = &B[B_inter_col_stride * j];
            C_col = &C[C_inter_col_stride * j];
            if (fixed_nnz_per_col) {
                RandBLAS::sparse_data::csc::apply_regular_csc_to_vector_from_left_ki<T>(
                    A_vals.data(), A_rows.data(), A_colptr[1],
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                );
            } else {
                RandBLAS::sparse_data::csc::apply_csc_to_vector_from_left_ki<T>(
                    A_vals.data(), A_rows.data(), A_colptr.data(),
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                ); 
            }
        }
    }
    return;
}


} // end namespace

#endif
