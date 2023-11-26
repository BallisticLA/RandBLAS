#ifndef randblas_sparse_skops_hh
#define randblas_sparse_skops_hh

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/util.hh"

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

namespace RandBLAS {
// =============================================================================
/// A distribution over sparse matrices.
///
struct SparseDist {

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many columns.
    const int64_t n_cols;

    // ---------------------------------------------------------------------------
    ///  If this distribution is short-axis major, then matrices sampled from
    ///  it will have exactly \math{\texttt{vec_nnz}} nonzeros per short-axis
    ///  vector (i.e., per column of a wide matrix or row of a tall matrix).
    //// One would be paranoid to set this higher than, say, eight, even when
    ///  sketching *very* high-dimensional data.
    ///
    ///  If this distribution is long-axis major, then matrices sampled from it
    ///  will have *at most* \math{\texttt{vec_nnz}} nonzeros per long-axis
    ///  vector (i.e., per row of a wide matrix or per column of a tall matrix).
    ///
    const int64_t vec_nnz;

    // ---------------------------------------------------------------------------
    ///  Constrains the sparsity pattern of matrices drawn from this distribution. 
    ///
    ///  Having major_axis==Short results in sketches are more likely to contain
    ///  useful geometric information, without making assumptions about the data
    ///  being sketched.
    ///
    const MajorAxis major_axis = MajorAxis::Short;
};

// =============================================================================
/// A sample from a prescribed distribution over sparse matrices.
///
template <typename T, typename RNG = r123::Philox4x32>
struct SparseSkOp {

    // ---------------------------------------------------------------------------
    ///  The distribution from which this sketching operator is sampled.
    ///  This member specifies the number of rows and columns of the sketching
    ///  operator.
    const SparseDist dist;

    // ---------------------------------------------------------------------------
    ///  The state that should be passed to the RNG when the full sketching 
    ///  operator needs to be sampled from scratch. 
    const RNGState<RNG> seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used by the next call to an RNG *after* the
    ///  full sketching operator has been sampled.
    RNGState<RNG> next_state;

    // ---------------------------------------------------------------------------
    /// We need workspace to store a representation of the sampled sketching
    /// operator. This member indicates who is responsible for allocating and 
    /// deallocating this workspace. If own_memory is true, then 
    /// RandBLAS is responsible.
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

    // ---------------------------------------------------------------------------
    ///
    /// @param[in] dist
    ///     A SparseDist object.
    ///     - Defines the number of rows and columns in this sketching operator.
    ///     - Indirectly controls sparsity pattern.
    ///     - Directly controls sparsity level.
    ///
    /// @param[in] state
    ///     An RNGState object.
    ///     - The RNG will use this as the starting point to generate all 
    ///       random numbers needed for this sketching operator.
    ///
    /// @param[in] rows
    ///     Pointer to int64_t array.
    ///     - stores row indices as part of the COO format.
    ///
    /// @param[in] cols
    ///     Pointer to int64_t array.
    ///     - stores column indices as part of the COO format.
    ///
    /// @param[in] vals
    ///     Pointer to array of real numerical type T.
    ///     - stores nonzeros as part of the COO format.
    ///
    SparseSkOp(
        SparseDist dist,
        const RNGState<RNG> &state,
        int64_t *rows,
        int64_t *cols,
        T *vals 
    );

    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        int64_t *rows,
        int64_t *cols,
        T *vals 
    ) : SparseSkOp(dist, RNGState<RNG>(key), rows, cols, vals) {};

    SparseSkOp(
        SparseDist dist,
        const RNGState<RNG> &state
    );

    SparseSkOp(
        SparseDist dist,
        uint32_t key
    ) : SparseSkOp(dist, RNGState<RNG>(key)) {};


    //  Destructor
    ~SparseSkOp();
};


template <typename T, typename RNG>
SparseSkOp<T,RNG>::SparseSkOp(
    SparseDist dist,
    const RNGState<RNG> &state
) :  // variable definitions
    dist(dist),
    seed_state(state),
    own_memory(true)
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    randblas_require(this->dist.vec_nnz > 0);
    // actual work
    int64_t minor_ax_len;
    if (this->dist.major_axis == MajorAxis::Short) {
        minor_ax_len = MAX(this->dist.n_rows, this->dist.n_cols);
    } else { 
        minor_ax_len = MIN(this->dist.n_rows, this->dist.n_cols);
    }
    int64_t nnz = this->dist.vec_nnz * minor_ax_len;
    this->rows = new int64_t[nnz];
    this->cols = new int64_t[nnz];
    this->vals = new T[nnz];
};

template <typename T, typename RNG>
SparseSkOp<T,RNG>::SparseSkOp(
    SparseDist dist,
    const RNGState<RNG> &state,
    int64_t *rows,
    int64_t *cols,
    T *vals
) :  // variable definitions
    dist(dist),
    seed_state(state),
    own_memory(false)
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    randblas_require(this->dist.vec_nnz > 0);
    // actual work
    this->rows = rows;
    this->cols = cols;
    this->vals = vals;
};

template <typename T, typename RNG>
SparseSkOp<T,RNG>::~SparseSkOp() {
    if (this->own_memory) {
        delete [] this->rows;
        delete [] this->cols;
        delete [] this->vals;
    }
};

// =============================================================================
/// Performs the work in sampling S from its underlying distribution. This 
/// entails populating S.rows, S.cols, and S.vals with COO-format sparse matrix
/// data.
///
/// RandBLAS will automatically call this function if and when it is needed.
///
/// @param[in] S
///     SparseSkOp object.
///
/// @return
///     An RNGState object. This is the state that should be used the next 
///     time the program needs to generate random numbers for a randomized
///     algorithm.
///     
template <typename T, typename RNG>
RNGState<RNG> fill_sparse(
    SparseSkOp<T,RNG> & S
) {
    int64_t long_ax_len = MAX(S.dist.n_rows, S.dist.n_cols);
    int64_t short_ax_len = MIN(S.dist.n_rows, S.dist.n_cols);

    bool is_wide = S.dist.n_rows == short_ax_len;
    int64_t *short_ax_idxs = (is_wide) ? S.rows : S.cols;
    int64_t *long_ax_idxs = (is_wide) ? S.cols : S.rows;

    if (S.dist.major_axis == MajorAxis::Short) {
        S.next_state = repeated_fisher_yates(
            S.seed_state, S.dist.vec_nnz, short_ax_len, long_ax_len,
            short_ax_idxs, long_ax_idxs, S.vals
        );
    } else {
        S.next_state = repeated_fisher_yates(
            S.seed_state, S.dist.vec_nnz, long_ax_len, short_ax_len,
            long_ax_idxs, short_ax_idxs, S.vals
        );
    }
    return S.next_state;
}

template <typename SKOP>
void print_sparse(SKOP const& S0) {
    std::cout << "SparseSkOp information" << std::endl;
    int64_t nnz;
    if (S0.dist.major_axis == MajorAxis::Short) {
        nnz = S0.dist.vec_nnz * MAX(S0.dist.n_rows, S0.dist.n_cols);
        std::cout << "\tSASO: short-axis-sparse operator" << std::endl;
    } else {
        nnz = S0.dist.vec_nnz * MIN(S0.dist.n_rows, S0.dist.n_cols);
        std::cout << "\tLASO: long-axis-sparse operator" << std::endl;
    }
    std::cout << "\tn_rows = " << S0.dist.n_rows << std::endl;
    std::cout << "\tn_cols = " << S0.dist.n_cols << std::endl;
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

// =============================================================================
/// WARNING: this function is not part of the public API.
///
template <typename T, typename RNG>
static auto repeated_fisher_yates(
    const RNGState<RNG> &state,
    int64_t vec_nnz,
    int64_t dim_major,
    int64_t dim_minor,
    int64_t *idxs_major,
    int64_t *idxs_minor,
    T *vals
) {
    randblas_error_if(vec_nnz > dim_major);
    std::vector<int64_t> vec_work(dim_major);
    for (int64_t j = 0; j < dim_major; ++j)
        vec_work[j] = j;
    std::vector<int64_t> pivots(vec_nnz);
    RNG gen;
    auto [ctr, key] = state;
    for (int64_t i = 0; i < dim_minor; ++i) {
        int64_t offset = i * vec_nnz;
        auto ctri = ctr;
        ctri.incr(offset);
        for (int64_t j = 0; j < vec_nnz; ++j) {
            // one step of Fisher-Yates shuffling
            auto rv = gen(ctri, key);
            int64_t ell = j + rv[0] % (dim_major - j);
            pivots[j] = ell;
            int64_t swap = vec_work[ell];
            vec_work[ell] = vec_work[j];
            vec_work[j] = swap;
            // update (rows, cols, vals)
            idxs_major[j + offset] = swap;
            vals[j + offset] = (rv[1] % 2 == 0) ? 1.0 : -1.0;
            idxs_minor[j + offset] = i;
            // increment counter
            ctri.incr();
        }
        // Restore vec_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes it easier to generate submatrices of
        //      a given SparseSkOp.
        for (int64_t j = 1; j <= vec_nnz; ++j) {
            int64_t jj = vec_nnz - j;
            int64_t swap = idxs_major[jj + offset];
            int64_t ell = pivots[jj];
            vec_work[jj] = vec_work[ell];
            vec_work[ell] = swap;
        }
        ctr = ctri;
    }
    return RNGState<RNG> {ctr, key};
}

} // end namespace RandBLAS

namespace RandBLAS::sparse {

using namespace RandBLAS;

template <typename SKOP>
static bool has_fixed_nnz_per_col(
    SKOP const& S0
) {
    if (S0.dist.major_axis == MajorAxis::Short) {
        return S0.dist.n_rows < S0.dist.n_cols;
    } else {
        return S0.dist.n_cols < S0.dist.n_rows;
    }
}

template <typename SKOP>
static int64_t nnz(
    SKOP const& S0
) {
    bool saso = S0.dist.major_axis == MajorAxis::Short;
    bool wide = S0.dist.n_rows < S0.dist.n_cols;
    if (saso & wide) {
        return S0.dist.vec_nnz * S0.dist.n_cols;
    } else if (saso & (!wide)) {
        return S0.dist.vec_nnz * S0.dist.n_rows;
    } else if (wide & (!saso)) {
        return S0.dist.vec_nnz * S0.dist.n_rows;
    } else {
        // tall LASO
        return S0.dist.vec_nnz * S0.dist.n_cols;
    }
}


// =============================================================================
/// WARNING: this function is not part of the public API.
///
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

// =============================================================================
/// WARNING: this function is not part of the public API.
///
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

// =============================================================================
/// WARNING: this function is not part of the public API.
///
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

// =============================================================================
/// WARNING: this function is not part of the public API.
///
template <typename T, typename SKOP>
static void apply_cscoo_csroo_left(
    T alpha,
    blas::Layout layout_A,
    blas::Layout layout_B,
    int64_t d,
    int64_t n,
    int64_t m,
    SKOP & S0,
    int64_t row_offset,
    int64_t col_offset,
    const T *A,
    int64_t lda,
    T *B,
    int64_t ldb
) {
    int64_t S_nnz;
    int64_t S0_nnz = nnz(S0);
    std::vector<int64_t> S_rows(S0_nnz, 0.0);
    std::vector<int64_t> S_cols(S0_nnz, 0.0);
    std::vector<T> S_vals(S0_nnz, 0.0);

    bool S0_fixed_nnz_per_col = has_fixed_nnz_per_col(S0);
    bool S_fixed_nnz_per_col = S0_fixed_nnz_per_col && (row_offset == 0) && (d == S0.dist.n_rows);

    if (S0_fixed_nnz_per_col) {
        S_nnz = filter_regular_cscoo<T>(
            S0.rows, S0.cols, S0.vals, S0.dist.vec_nnz,
            col_offset, col_offset + m,
            row_offset, row_offset + d,
            S_rows.data(), S_cols.data(), S_vals.data()
        );
    } else {
        S_nnz = filter_and_convert_regular_csroo_to_cscoo<T>(
            S0.rows, S0.cols, S0.vals, S0.dist.vec_nnz,
            col_offset, col_offset + m,
            row_offset, row_offset + d,
            S_rows.data(), S_cols.data(), S_vals.data()
        );
    }
    blas::scal<T>(S_nnz, alpha, S_vals.data(), 1);

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

    if (S_fixed_nnz_per_col) {
        S_nnz = -S0.dist.vec_nnz; // this value in interpreted differently when negative.
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
                S_rows.data(), S_cols.data(), S_vals.data(), m, S_nnz
            );
        }
    }
    return;
}

// =============================================================================
/// Return a SparseSkOp object representing the transpose of S.
///
/// @param[in] S
///     SparseSkOp object.
/// @return 
///     A new SparseSkOp object that depends on the memory underlying S.
///     (In particular, it depends on S.rows, S.cols, and S.vals.)
///     
template <typename SKOP>
static auto transpose(SKOP const& S) {
    SparseDist dist = {
        .n_rows = S.dist.n_cols,
        .n_cols = S.dist.n_rows,
        .vec_nnz = S.dist.vec_nnz,
        .major_axis = S.dist.major_axis
    };
    SKOP St(dist, S.seed_state, S.cols, S.rows, S.vals);
    St.next_state = S.next_state;
    return St;
}


// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |submat| mathmacro:: \operatorname{submat}
///   .. |lda| mathmacro:: \mathrm{lda}
///   .. |ldb| mathmacro:: \mathrm{ldb}
///   .. |opA| mathmacro:: \mathrm{opA}
///   .. |opS| mathmacro:: \mathrm{opS}
///
/// @endverbatim
/// LSKGES: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sparse sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(d, m, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, d, m)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{i_off}, \texttt{j_off})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}
///     - The number of rows in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(A))}.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of columns in \math{\op(\submat(S))}
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] S
///    A SparseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] i_off
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{i_off}, :]}.
///
/// @param[in] j_off
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{j_off}]}. 
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename SKOP>
void lskges(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // \op(A) is m-by-n
    int64_t m, // \op(S) is d-by-m
    T alpha,
    SKOP &S,
    int64_t row_offset,
    int64_t col_offset,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    if (S.rows == nullptr|| S.cols == nullptr || S.vals == nullptr)
        fill_sparse(S);

    // handle applying a transposed sparse sketching operator.
    if (opS == blas::Op::Trans) {
        auto St = transpose(S);
        lskges(
            layout, blas::Op::NoTrans, opA,
            d, n, m, alpha, St, col_offset, row_offset,
            A, lda, beta, B, ldb
        );
        return; 
    }
    // Below this point, we can assume S is not transposed.
    randblas_require(S.dist.n_rows >= d);
    randblas_require(S.dist.n_cols >= m);

    // Dimensions of A, rather than \op(A)
    blas::Layout layout_B = layout;
    blas::Layout layout_opA;
    int64_t rows_A, cols_A;
    if (opA == blas::Op::NoTrans) {
        rows_A = m;
        cols_A = n;
        layout_opA = layout;
    } else {
        rows_A = n;
        cols_A = m;
        layout_opA = (layout == blas::Layout::ColMajor) ? blas::Layout::RowMajor : blas::Layout::ColMajor;
    }

    // Check dimensions and compute B = beta * B.
    //      Note: both A and B are checked based on "layout"; A is *not* checked on layout_opA.
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(d, beta, &B[i*ldb], 1);
    } else {
        randblas_require(ldb >= n);
        randblas_require(lda >= cols_A);
        for (int64_t i = 0; i < d; ++i)
            RandBLAS::util::safe_scal(n, beta, &B[i*ldb], 1);
    }

    // Perform the sketch
    if (alpha != 0)
        apply_cscoo_csroo_left(alpha, layout_opA, layout_B, d, n, m, S, row_offset, col_offset, A, lda, B, ldb);
    return;
}


// =============================================================================
/// RSKGES: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(\submat(S))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sparse sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(m, d, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, n, d)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{i_off}, \texttt{j_off})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}.
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\op(\mat(A))}
///     - The number of rows in \math{\op(\submat(S))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] S
///    A SparseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] i_off
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{i_off}, :]}.
///
/// @param[in] j_off
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{j_off}]}. 
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename RNG>
void rskges(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(S) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SparseSkOp<T,RNG> &S0,
    int64_t i_off,
    int64_t j_off,
    T beta,
    T *B,
    int64_t ldb
) { 
    //
    // Compute B = op(A) op(submat(S)) by reduction to LSKGES. We start with
    //
    //      B^T = op(submat(S))^T op(A)^T.
    //
    // Then we interchange the operator "op(*)" in op(A) and (*)^T:
    //
    //      B^T = op(submat(S))^T op(A^T).
    //
    // We tell LSKGES to process (B^T) and (A^T) in the opposite memory layout
    // compared to the layout for (A, B).
    // 
    using blas::Layout;
    using blas::Op;
    auto trans_opS = (opS == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    lskges(
        trans_layout, trans_opS, opA,
        d, m, n, alpha, S0, i_off, j_off, A, lda, beta, B, ldb
    );
}

} // end namespace RandBLAS::sparse_ops

#endif