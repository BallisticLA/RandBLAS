#ifndef randblas_sparse_data_coo
#define randblas_sparse_data_coo
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

enum class NonzeroSort : char {
    CSC = 'C',
    CSR = 'R',
    None = 'N'
};

NonzeroSort coo_sort_type(int64_t nnz, int64_t *rows, int64_t *cols) {
    bool csr_okay = true;
    bool csc_okay = true;
    auto increasing_by_csr = [](int64_t i0, int64_t j0, int64_t i1, int64_t j1) {
        if (i0 > i1) {
            return false;
        } else if (i0 == i1) {
            return j0 <= j1;
        } else {
            return true;
        }
    };
    auto increasing_by_csc = [](int64_t i0, int64_t j0, int64_t i1, int64_t j1) {
        if (j0 > j1) {
            return false;
        } else if (j0 == j1) {
            return i0 <= i1;
        } else {
            return true;
        }
    };
    for (int64_t ell = 1; ell < nnz; ++ell) {
        auto i0 = rows[ell-1];
        auto j0 = cols[ell-1];
        auto i1 = rows[ell];
        auto j1 = cols[ell];
        if (csr_okay) {
            csr_okay = increasing_by_csr(i0, j0, i1, j1);
        }
        if (csc_okay) {
            csc_okay = increasing_by_csc(i0, j0, i1, j1);
        }
        if (!csr_okay && !csc_okay)
            break;
    }
    if (csr_okay) {
        return NonzeroSort::CSR;
    } else if (csc_okay) {
        return NonzeroSort::CSC;
    } else {
        return NonzeroSort::None;
    }
    
}


template <typename T>
struct COOMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const IndexBase index_base;
    const bool own_memory;
    int64_t nnz;
    T *vals;
    int64_t *rows;
    int64_t *cols;
    NonzeroSort sort;
    int64_t *_sortptr;
    // bool _self_allocate_sortptr = true;
    bool _can_reserve = true;

    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(true) {
        this->nnz = 0;
        this->vals = nullptr;
        this->rows = nullptr;
        this->cols = nullptr;
        this->sort = NonzeroSort::None;
        // this->_self_allocate_sort_ptr = true;
        // this->_sortptr = new int64_t[MAX(n_rows, n_cols) + 1];
    };

    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        int64_t *rows,
        int64_t *cols,
        bool compute_sort_type = true,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(false) {
        this->nnz = nnz;
        this->vals = vals;
        this->rows = rows;
        this->cols = cols;
        if (compute_sort_type) {
            this->sort = coo_sort_type(nnz, rows, cols);
        } else {
            this->sort = NonzeroSort::None;
        }
    };

    ~COOMatrix() {
        if (this->own_memory) {
            delete [] this->vals;
            delete [] this->rows;
            delete [] this->cols;
           
        }
    };

    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->nnz = nnz;
        this->vals = new T[nnz];
        this->rows = new int64_t[nnz];
        this->cols = new int64_t[nnz];
        this->_can_reserve = false;
    }

};

template <typename T>
void sort_coo_data(
    NonzeroSort s,
    int64_t nnz,
    T *vals,
    int64_t *rows,
    int64_t *cols
) {
    if (s == NonzeroSort::None)
        return;
    // note: this implementation makes unnecessary copies

    // get a vector-of-triples representation of the matrix
    using tuple_type = std::tuple<int64_t, int64_t, T>;
    std::vector<tuple_type> nonzeros;
    nonzeros.reserve(nnz);
    for (int64_t ell = 0; ell < nnz; ++ell)
        nonzeros.emplace_back(rows[ell], cols[ell], vals[ell]);

    // sort the vector-of-triples representation
    auto sort_func = [s](tuple_type const& t1, tuple_type const& t2) {
        if (s == NonzeroSort::CSR) {
            if (std::get<0>(t1) < std::get<0>(t2)) {
                return true;
            } else if (std::get<0>(t1) > std::get<0>(t2)) {
                return false;
            } else if (std::get<1>(t1) < std::get<1>(t2)) {
                return true;
            } else {
                return false;
            }
        } else {
            if (std::get<1>(t1) < std::get<1>(t2)) {
                return true;
            } else if (std::get<1>(t1) > std::get<1>(t2)) {
                return false;
            } else if (std::get<0>(t1) < std::get<0>(t2)) {
                return true;
            } else {
                return false;
            }
        }
    };
    std::sort(nonzeros.begin(), nonzeros.end(), sort_func);

    // unpack the vector-of-triples rep into the triple-of-vectors rep
    for (int64_t ell = 0; ell < nnz; ++ell) {
        tuple_type tup = nonzeros[ell];
        vals[ell] = std::get<2>(tup);
        rows[ell] = std::get<0>(tup);
        cols[ell] = std::get<1>(tup);
    }
    return;
}

template <typename T>
void sort_coo_data(
    NonzeroSort s,
    COOMatrix<T> &spmat
) {
    sort_coo_data(s, spmat.nnz, spmat.vals, spmat.rows, spmat.cols);
    spmat.sort = s;
    return;
}

template <typename T>
static auto transpose(COOMatrix<T> &S) {
    COOMatrix<T> St(S.n_rows, S.n_cols, S.nnz, S.vals, S.cols, S.rows, false, S.index_base);
    if (S.sort == NonzeroSort::CSC) {
        St.sort = NonzeroSort::CSR;
    } else if (S.sort == NonzeroSort::CSR) {
        St.sort = NonzeroSort::CSC;
    }
    return St;
}


} // end namespace RandBLAS::sparse_data


namespace RandBLAS::sparse_data::coo {

using namespace RandBLAS::sparse_data;

static inline void set_filtered_colptr(
    int64_t len_colidxs,
    const int64_t *colidxs,
    int64_t col_start,
    int64_t col_end,
    int64_t *new_colptr
) {
    int64_t ell;
    // check that colidxs is sorted in increasing order
    for (ell = 1; ell < len_colidxs; ++ell)
        randblas_require(colidxs[ell-1] <= colidxs[ell]);
    // compress filter and compress colidxs into colptr
    int64_t prev_col = col_start - 1;
    int64_t curr_col, j, colptr_update_limit;
    for (ell = 0; ell < len_colidxs; ++ell) {
        curr_col = colidxs[ell];
        if (curr_col < col_start)
            continue;
        colptr_update_limit = std::min(curr_col, col_end);
        for (j = prev_col + 1; j <= colptr_update_limit; ++j)
            new_colptr[j - col_start] = ell;
        prev_col = curr_col;
    }
    return;
}

template <typename T>
static int64_t set_filtered_csc_from_cscoo(
    // COO-format matrix data, in CSC order
    const T       *vals,
    const int64_t *rowidxs,
    const int64_t *colidxs,
    int64_t nnz,
    // submatrix bounds
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end,
    // CSC-format data for the submatrix
    T       *new_vals,
    int64_t *new_rowidxs,
    int64_t *new_colptr
) {
    int64_t new_nnz = 0;
    int64_t i, j, k;
    set_filtered_colptr(nnz, colidxs, col_start, col_end, new_colptr);
    for (j = 0; j < col_end - col_start; ++j) {
        for (k = new_colptr[j]; k < new_colptr[j+1]; ++k) {
            i = rowidxs[k];
            if (i < row_start)
                continue;
            if (i >= row_end)
                break;
            new_vals[new_nnz] = vals[k];
            new_rowidxs[new_nnz] = i - row_start;
            new_nnz += 1;
        }
    }
    return new_nnz;
}

template <typename T>
static void apply_csc_to_vector_from_left(
    // CSC-format data
    int64_t n_cols,
    const T *vals,
    int64_t *rowidxs,
    int64_t *colptr,
    // input-output vector data
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    int64_t i = 0;
    for (int64_t c = 0; c < n_cols; ++c) {
        T scale = v[c * incv];
        while (i < colptr[c+1]) {
            int64_t row = rowidxs[i];
            Av[row * incAv] += (vals[i] * scale);
            i += 1;
        }
    }
}



// =============================================================================
/// WARNING: this function is not part of the public API.
///
template <typename T>
static void apply_coo_left(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    COOMatrix<T> & A0,
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
        apply_cscoo_submat_to_vector_from_left(layout_B, layout_C, d, n, m, A0, row_offset, col_offset, B, ldb, C, ldc);
        sort_coo_data(orig_sort, A0);
        return;
    }

    // Step 2: make a CSC-sort-order COOMatrix that represents the desired submatrix of S.
    //      While we're at it, reduce to the case when alpha = 1.0 by scaling the values
    //      of the matrix we just created.
    int64_t A_nnz;
    int64_t A0_nnz = A0.nnz;
    std::vector<int64_t> A_rows(A0_nnz, 0);
    std::vector<int64_t> A_colptr(m + 1, 0);
    std::vector<T> A_vals(A0_nnz, 0.0);
    A_nnz = set_filtered_csc_from_cscoo(
        A0.vals, A0.rows, A0.cols, A0.nnz,
        col_offset, col_offset + m,
        row_offset, row_offset + d,
        A_vals.data(), A_rows.data(), A_colptr.data()
    );
    blas::scal<T>(A_nnz, alpha, A_vals.data(), 1);


    // Step 3: Apply "S" to the left of A to get B += S*A.
    int64_t B_inter_col_stride, B_intra_col_stride;
    if (layout_B == blas::Layout::ColMajor) {
        B_inter_col_stride = ldb;
        B_intra_col_stride = 1;
    } else {
        B_inter_col_stride = 1;
        B_intra_col_stride = ldb;
    }
    int64_t C_inter_col_stride, C_intra_col_stride;
    if (layout_C == blas::Layout::ColMajor) {
        C_inter_col_stride = ldc;
        C_intra_col_stride = 1;
    } else {
        C_inter_col_stride = 1;
        C_intra_col_stride = ldc;
    }

    #pragma omp parallel default(shared)
    {
        const T *B_col = nullptr;
        T *C_col = nullptr;
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < n; k++) {
            B_col = &B[B_inter_col_stride * k];
            C_col = &C[C_inter_col_stride * k];
            apply_csc_to_vector_from_left<T>(
                m, A_vals.data(), A_rows.data(), A_colptr.data(),
                B_col, B_intra_col_stride,
                C_col, C_intra_col_stride
            );
        }
    }
    return;
}

} // end namespace RandBLAS::sparse_data::coo

#endif
