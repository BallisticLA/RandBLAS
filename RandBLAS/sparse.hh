#ifndef randblas_sparse_hh
#define randblas_sparse_hh

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"

#include <blas.hh>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace RandBLAS::sparse {

using namespace RandBLAS::base;

enum class SparseDistName : char {
    SASO = 'S',      // short-axis-sparse operator
    LASO = 'L'       // long-axis-sparse operator
};

struct SparseDist {
    const SparseDistName family = SparseDistName::SASO;
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
};

template <typename T, typename RNG = r123::Philox4x32>
struct SparseSkOp {
    const SparseDist dist;
    const base::RNGState<RNG> seed_state; // maybe "self_seed"
    base::RNGState<RNG> next_state; // maybe "next_seed"
    const bool own_memory = true;
    
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    int64_t *rows = nullptr;
    int64_t *cols = nullptr;
    T *vals = nullptr;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    SparseSkOp(
        SparseDist dist_,
        const base::RNGState<RNG> &state_,
        int64_t *rows_ = nullptr,
        int64_t *cols_ = nullptr,
        T *vals_ = nullptr 
    );

    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows,
        int64_t *cols,
        T *vals 
    ) : SparseSkOp(dist, {{{ctr_offset}}, {{key}}}, rows, cols, vals) {};
    
    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        int64_t vec_nnz,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows = nullptr,
        int64_t *cols = nullptr,
        T *vals = nullptr 
    ) : SparseSkOp({family, n_rows, n_cols, vec_nnz},
        key, ctr_offset, rows, cols, vals) {};

    //  Destructor
    ~SparseSkOp();
};

// Implementation of elementary constructor
template <typename T, typename RNG>
SparseSkOp<T,RNG>::SparseSkOp(
    SparseDist dist_,
    const base::RNGState<RNG> &state_,
    int64_t *rows_,
    int64_t *cols_,
    T *vals_
) :  // variable definitions
    dist(dist_),
    seed_state(state_),
    own_memory(!rows_ && !cols_ && !vals_)
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    randblas_require(this->dist.vec_nnz > 0);
    // Initialization logic
    //
    //      own_memory is a bool that's true iff the
    //      rows_, cols_, and vals_ pointers were all nullptr.
    //
    int64_t rep_ax_len;
    if (this->dist.family == SparseDistName::SASO) {
        rep_ax_len = MAX(this->dist.n_rows, this->dist.n_cols);
    } else { 
        rep_ax_len = MIN(this->dist.n_rows, this->dist.n_cols);
    }
    if (this->own_memory) {
        int64_t nnz = this->dist.vec_nnz * rep_ax_len;
        this->rows = new int64_t[nnz];
        this->cols = new int64_t[nnz];
        this->vals = new T[nnz];
    } else {
        randblas_require(rows_ && cols_ && vals_);
        //  If any of rows_, cols_, and vals_ are not nullptr,
        //  then none of them are nullptr.
        this->rows = rows_;
        this->cols = cols_;
        this->vals = vals_;
    }
};

template <typename T, typename RNG>
SparseSkOp<T,RNG>::~SparseSkOp() {
    if (this->own_memory) {
        delete [] this->rows;
        delete [] this->cols;
        delete [] this->vals;
    }
};

template <typename SKOP>
static bool has_fixed_nnz_per_col(
    SKOP const& S0
) {
    if (S0.dist.family == SparseDistName::SASO) {
        return S0.dist.n_rows < S0.dist.n_cols;
    } else {
        return S0.dist.n_cols < S0.dist.n_rows;
    }
}

template <typename T, typename RNG>
static
auto repeated_fisher_yates(
    const RNGState<RNG> &state,
    int64_t num_vecs,
    int64_t vec_len,
    int64_t vec_nnz,
    int64_t *vec_ax_idxs,
    int64_t *rep_ax_idxs,
    T *vals
) {
    std::vector<int64_t> vec_work(vec_len);
    for (int64_t j = 0; j < vec_len; ++j)
        vec_work[j] = j;
    std::vector<int64_t> pivots(vec_nnz);
    RNG gen;
    auto [ctr, key] = state;
    for (int64_t i = 0; i < num_vecs; ++i) {
        int64_t offset = i * vec_nnz;
        auto ctri = ctr;
        ctri.incr(offset);
        for (int64_t j = 0; j < vec_nnz; ++j) {
            // one step of Fisher-Yates shuffling
            auto rv = gen(ctri, key);
            int64_t ell = j + rv[0] % (vec_len - j);
            pivots[j] = ell;
            int64_t swap = vec_work[ell];
            vec_work[ell] = vec_work[j];
            vec_work[j] = swap;
            // update (rows, cols, vals)
            vec_ax_idxs[j + offset] = swap;
            vals[j + offset] = (rv[1] % 2 == 0) ? 1.0 : -1.0;
            rep_ax_idxs[j + offset] = i;
            // increment counter
            ctri.incr();
        }
        // Restore vec_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes it easier to generate submatrices of
        //      a given SparseSkOp.
        for (int64_t j = 1; j <= vec_nnz; ++j) {
            int64_t jj = vec_nnz - j;
            int64_t swap = vec_ax_idxs[jj + offset];
            int64_t ell = pivots[jj];
            vec_work[jj] = vec_work[ell];
            vec_work[ell] = swap;
        }
        ctr = ctri;
    }
    return RNGState<RNG> {ctr, key};
}

template <typename SKOP>
static
auto fill_sparse(
    SKOP & S0
) {
    int64_t long_ax_len = MAX(S0.dist.n_rows, S0.dist.n_cols);
    int64_t short_ax_len = MIN(S0.dist.n_rows, S0.dist.n_cols);

    bool is_wide = S0.dist.n_rows == short_ax_len;
    int64_t *short_ax_idxs = (is_wide) ? S0.rows : S0.cols;
    int64_t *long_ax_idxs = (is_wide) ? S0.cols : S0.rows;

    int64_t *vec_ax_idxs, *rep_ax_idxs;
    int64_t vec_len, num_vecs;
    if (S0.dist.family == SparseDistName::SASO) {
        vec_len = short_ax_len;
        num_vecs = long_ax_len;
        vec_ax_idxs = short_ax_idxs;
        rep_ax_idxs = long_ax_idxs;
    } else {
        vec_len = long_ax_len;
        num_vecs = short_ax_len;
        vec_ax_idxs = long_ax_idxs;
        rep_ax_idxs = short_ax_idxs;
    }
    S0.next_state = repeated_fisher_yates(
        S0.seed_state, num_vecs, vec_len,
        S0.dist.vec_nnz, vec_ax_idxs, rep_ax_idxs, S0.vals
    );
    return S0.next_state;
}

template <typename SKOP>
void print_saso(SKOP const& S0) {
    std::cout << "SparseSkOp information" << std::endl;
    std::cout << "\tn_rows = " << S0.dist.n_rows << std::endl;
    std::cout << "\tn_cols = " << S0.dist.n_cols << std::endl;
    int64_t nnz;
    if (S0.dist.family == SparseDistName::SASO) {
        nnz = S0.dist.vec_nnz * MAX(S0.dist.n_rows, S0.dist.n_cols);
    } else {
        nnz = S0.dist.vec_nnz * MIN(S0.dist.n_rows, S0.dist.n_cols);
    }
    std::cout << "\tvector of row indices\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << S0.rows[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of column indices\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << S0.cols[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of values\n\t\t";
    for (int64_t i = 0; i < nnz; ++i) {
        std::cout << S0.vals[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
static int64_t filter_regular_cscoo(
    const int64_t *nzidx2row,
    const int64_t *nzidx2col,
    const T *nzidx2val,
    int64_t vec_nnz,
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end,
    int64_t *rows_view,
    int64_t *cols_view,
    T *vals_view
) {
    // (nzidx2row, nzidx2col, nzidx2val) define a sparse matrix S0
    //  in COO format with a CSC-like property. The CSC-like property
    //  is that all nonzero entries in a given column of S0 are 
    //  contiguous in "nzidx2col". The sparse matrix S0 must be "regular"
    //  in the sense that it must have exactly vec_nnz nonzeros in each
    //  column.
    //
    //  This function writes data to (rows_view, cols_view, vals_view)
    //  for a sparse matrix S in the same "COO/CSC-like" format.
    //  Mathematically, we have S = S0[row_start:row_end, col_start:col_end].
    //  The sparse matrix S does not necessarily have a fixed number of
    //  nonzeros in each column.
    //
    //  Neither S0 nor S need to be wide.  
    //
    int64_t nnz = 0;
    for (int64_t i = col_start * vec_nnz; i < col_end * vec_nnz; ++i) {
        int64_t row = nzidx2row[i];
        if (row_start <= row && row < row_end)  {
            rows_view[nnz] = row - row_start;
            cols_view[nnz] = nzidx2col[i] - col_start;
            vals_view[nnz] = nzidx2val[i];
            nnz += 1;
        }
    }
    return nnz;
}

template <typename T>
static int64_t filter_and_convert_regular_csroo_to_cscoo(
    const int64_t *nonzeroidx2row,
    const int64_t *nonzeroidx2col,
    const T *nonzeroidx2val,
    int64_t vec_nnz,
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end,
    int64_t *rows_view,
    int64_t *cols_view,
    T *vals_view
) {
    // (nonzeroidx2row, nonzeroidx2col, nonzeroidx2val) define a sparse matrix S0
    //  in COO format with a CSR-like property. The CSR-like property
    //  is that all nonzero entries in a given row of S0 are 
    //  contiguous in "nonzeroidx2col". The sparse matrix S0 must be "regular"
    //  in the sense that it must have exactly vec_nnz nonzeros in each
    //  row.
    //
    //  This function writes data to (rows_view, cols_view, vals_view)
    //  for a sparse matrix S in an analogous "COO/CSC-like" format.
    //  Mathematically, we have S = S0[row_start:row_end, col_start:col_end].
    //  The sparse matrix S does not necessarily have a fixed number of
    //  nonzeros in each column.
    //
    //  Neither S0 nor S need to be wide.  
    //
    using tuple_type = std::tuple<int64_t, int64_t, T>;
    std::vector<tuple_type> nonzeros;
    nonzeros.reserve((row_end - row_start) * vec_nnz);

    for (int64_t i = row_start * vec_nnz; i < row_end * vec_nnz; ++i) {
        int64_t col = nonzeroidx2col[i];
        if (col_start <= col && col < col_end)  {
            nonzeros.emplace_back(
                nonzeroidx2row[i] - row_start,
                col - col_start,
                nonzeroidx2val[i]);
        }
    }

    auto sort_func = [](tuple_type const& t1, tuple_type const& t2) {
        if (std::get<1>(t1) < std::get<1>(t2)) {
            return true;
        } else if (std::get<1>(t1) > std::get<1>(t2)) {
            return false;
        }
        // t1.second == t2.second
        if (std::get<0>(t1) < std::get<0>(t2)) {
            return true;
        } else {
            return false;
        }
    };
    std::sort(nonzeros.begin(), nonzeros.end(), sort_func);
    
    int64_t nnz = nonzeros.size();
    for (int64_t i = 0; i < nnz; ++i) {
        tuple_type tup = nonzeros[i];
        rows_view[i] = std::get<0>(tup);
        cols_view[i] = std::get<1>(tup);
        vals_view[i] = std::get<2>(tup);
    }
    return nnz;
}

template <typename T>
static void apply_cscoo_submat_to_vector_from_left(
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Sv,          // Sv += S * v.
    int64_t incSv,  // stride between elements of Sv
    const int64_t *rows_view,
    const int64_t *cols_view,
    const T       *vals_view,
    int64_t num_cols,
    int64_t nnz
) {
    // (rows_view, cols_view, vals_view) define a sparse matrix
    // "S" in COO-format with a CSC-like property.
    //
    //      The CSC-like property requires that all nonzero entries
    //      in a given column of the sparse matrix are contiguous in
    //      "cols_view" and that entries in cols_view are nondecreasing.
    //  
    //      The sparse matrix does not need to be wide.
    //
    // When nnz < 0, each of the "num_cols" blocks in "cols_view" 
    // must be the same size -nnz > 0. Furthermore, consecutive values
    // in cols_view can only differ from one another by at most one.
    //
    if (nnz >= 0) {
        int64_t i = 0;
        for (int64_t c = 0; c < num_cols; ++c) {
            T scale = v[c * incv];
            while (i < nnz && cols_view[i] == c) {
                int64_t row = rows_view[i];
                Sv[row * incSv] += (vals_view[i] * scale);
                i += 1;
            }
        }
    } else {
        int64_t col_nnz = -nnz;
        for (int64_t c = 0; c < num_cols; ++c) {
            T scale = v[c * incv];
            for (int64_t j = c * col_nnz; j < (c + 1) * col_nnz; ++j) {
                int64_t row = rows_view[j];
                Sv[row * incSv] += (vals_view[j] * scale);
            }
        }
    }
}

template <typename T, typename RNG>
static void apply_cscoo_csroo_left(
    blas::Layout layout_A,
    blas::Layout layout_B,
    int64_t d,
    int64_t n,
    int64_t m,
    SparseSkOp<T,RNG> & S0,
    int64_t row_offset,
    int64_t col_offset,
    const T *A,
    int64_t lda,
    T *B,
    int64_t ldb
) {
    int64_t vec_nnz = S0.dist.vec_nnz;
    int64_t nnz;
    int64_t *S_rows, *S_cols;
    T *S_vals;

    bool use_existing_memory = false;
    bool S0_fixed_nnz_per_col = has_fixed_nnz_per_col(S0);
    if (S0_fixed_nnz_per_col) {
        bool S_fixed_nnz_per_col = (row_offset == 0) && (d == S0.dist.n_rows);
        use_existing_memory = (col_offset == 0) && S_fixed_nnz_per_col;
        // ^ If col_offset is nonzero, then we need to make an altered 
        //   version of S0.cols that shifts all entries by -col_offset.
        if (use_existing_memory) {
            S_rows = S0.rows;
            S_cols = S0.cols;
            S_vals = S0.vals;
            // we set nnz in a moment.
        } else {
            S_rows = new int64_t[m * vec_nnz]{};
            S_cols = new int64_t[m * vec_nnz]{};
            S_vals = new       T[m * vec_nnz]{};
            nnz = filter_regular_cscoo<T>(
                S0.rows, S0.cols, S0.vals, vec_nnz,
                col_offset, col_offset + m,
                row_offset, row_offset + d,
                S_rows, S_cols, S_vals
            );
            // ^ we might overwrite that value of nnz.
        }
        if (S_fixed_nnz_per_col)
            nnz = -vec_nnz;
    } else {
        // We have to use new memory, because we need the CSCOO representation.
        S_rows = new int64_t[d * vec_nnz]{};
        S_cols = new int64_t[d * vec_nnz]{};
        S_vals = new       T[d * vec_nnz]{};
        nnz = filter_and_convert_regular_csroo_to_cscoo<T>(
            S0.rows, S0.cols, S0.vals, vec_nnz,
            col_offset, col_offset + m,
            row_offset, row_offset + d,
            S_rows, S_cols, S_vals
        );
    }
    // Once we have (S_rows, S_cols, S_vals) in the format ensured
    // by the functions above, we apply the resulting sparse matrix "S"
    // to the left of A to get B = S*A.
    //
    // This function does not require that S or S0 is wide.

    int64_t A_inter_col_stride, A_intra_col_stride;
    if (layout_A == blas::Layout::ColMajor) {
        A_inter_col_stride = lda;
        A_intra_col_stride = 1;
    } else {
        A_inter_col_stride = 1;
        A_intra_col_stride = lda;
    }
    int64_t B_inter_col_stride, B_intra_col_stride;
    if (layout_B == blas::Layout::ColMajor) {
        B_inter_col_stride = ldb;
        B_intra_col_stride = 1;
    } else {
        B_inter_col_stride = 1;
        B_intra_col_stride = ldb;
    }

    #pragma omp parallel default(shared)
    {
        const T *A_col = nullptr;
        T *B_col = nullptr;
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < n; k++) {
            A_col = &A[A_inter_col_stride * k];
            B_col = &B[B_inter_col_stride * k];
            apply_cscoo_submat_to_vector_from_left<T>(
                A_col, A_intra_col_stride,
                B_col, B_intra_col_stride,
                S_rows, S_cols, S_vals, m, nnz
            );
        }
    }

    if (!use_existing_memory) {
        delete [] S_rows;
        delete [] S_cols;
        delete [] S_vals;
    }
    return;
}

template <typename SKOP>
static
auto transpose(SKOP const& S0) {
    SparseDist dist = {
        .family = S0.dist.family,
        .n_rows = S0.dist.n_cols,
        .n_cols = S0.dist.n_rows,
        .vec_nnz = S0.dist.vec_nnz
    };
    SKOP S1(
        dist,
        S0.seed_state,
        S0.cols,
        S0.rows,
        S0.vals
    );
    S1.next_state = S0.next_state;
    return S1;
}

template <typename T, typename RNG>
void lskges(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    SparseSkOp<T,RNG> &S0,
    int64_t row_offset,
    int64_t col_offset,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    randblas_require(S0.rows != NULL); // must be filled.
    randblas_require(alpha == 1.0); // implementation limitation
    randblas_require(beta == 0.0); // implementation limitation

    // handle applying a transposed sparse sketching operator.
    if (transS == blas::Op::Trans) {
        auto S1 = transpose(S0);
        lskges(
            layout, blas::Op::NoTrans, transA,
            d, m, n, alpha, S1, col_offset, row_offset,
            A, lda, beta, B, ldb
        );
        return; 
    }
    // Below this point, we can assume S0 is not transposed.

    // Dimensions of A, rather than op(A)
    blas::Layout layout_B = layout;
    blas::Layout layout_A;
    int64_t rows_A, cols_A;
    if (transA == blas::Op::NoTrans) {
        rows_A = m;
        cols_A = n;
        layout_A = layout;
    } else {
        rows_A = n;
        cols_A = m;
        layout_A = (layout == blas::Layout::ColMajor) ? blas::Layout::RowMajor : blas::Layout::ColMajor;
        // ^ Lie.
    }

    // Dimensionality sanity checks
    //      Both A and B are checked based on "layout"; A is *not* checked on layout_A.
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }

    // Perform the sketch
    apply_cscoo_csroo_left(layout_A, layout_B, d, n, m, S0, row_offset, col_offset, A, lda, B, ldb);
    return;
}

} // end namespace RandBLAS::sparse_ops

#endif
