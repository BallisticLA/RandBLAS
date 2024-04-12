#ifndef randblas_sparse_skops_hh
#define randblas_sparse_skops_hh

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_data/spmm_dispatch.hh"

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

using RandBLAS::SignedInteger;

// =============================================================================
/// A sample from a prescribed distribution over sparse matrices.
///
template <typename T, typename RNG = r123::Philox4x32, SignedInteger sint_t = int64_t>
struct SparseSkOp {

    using RNG_t = RNG;
    using T_t = T;
    using index_t = sint_t;

    const int64_t n_rows;
    const int64_t n_cols;

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

    // ---------------------------------------------------------------------------
    /// A flag (indicating a sufficient condition) that the data underlying the
    /// sparse matrix has already been sampled.
    bool known_filled = false;
    
    
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    sint_t *rows = nullptr;
    sint_t *cols = nullptr;
    T *vals = nullptr;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    // ---------------------------------------------------------------------------
    // 
    //  @param[in] dist
    //      A SparseDist object.
    //      - Defines the number of rows and columns in this sketching operator.
    //      - Indirectly controls sparsity pattern.
    //      - Directly controls sparsity level.
    // 
    //  @param[in] state
    //      An RNGState object.
    //      - The RNG will use this as the starting point to generate all 
    //        random numbers needed for this sketching operator.
    // 
    //  @param[in] rows
    //      Pointer to int64_t array.
    //      - stores row indices as part of the COO format.
    // 
    //  @param[in] cols
    //      Pointer to int64_t array.
    //      - stores column indices as part of the COO format.
    // 
    //  @param[in] vals
    //      Pointer to array of real numerical type T.
    //      - stores nonzeros as part of the COO format.
    //  
    //  @param[in] known_filled
    //      A boolean. If true, then the arrays pointed to by
    //      (rows, cols, vals) already contain the randomly sampled
    //      data defining this sketching operator.
    //      
    SparseSkOp(
        SparseDist dist,
        const RNGState<RNG> &state,
        sint_t *rows,
        sint_t *cols,
        T *vals,
        bool known_filled = true
    );

    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        sint_t *rows,
        sint_t *cols,
        T *vals 
    ) : SparseSkOp(dist, RNGState<RNG>(key), rows, cols, vals) {};


    ///---------------------------------------------------------------------------
    /// The preferred constructor for SparseSkOp objects. There are other 
    /// constructors, but they don't appear in the web documentation.
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


template <typename T, typename RNG, SignedInteger sint_t>
SparseSkOp<T,RNG,sint_t>::SparseSkOp(
    SparseDist dist,
    const RNGState<RNG> &state
) :  // variable definitions
    n_rows(dist.n_rows),
    n_cols(dist.n_cols),
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
    this->rows = new sint_t[nnz];
    this->cols = new sint_t[nnz];
    this->vals = new T[nnz];
};

template <typename T, typename RNG, SignedInteger sint_t>
SparseSkOp<T,RNG,sint_t>::SparseSkOp(
    SparseDist dist,
    const RNGState<RNG> &state,
    sint_t *rows,
    sint_t *cols,
    T *vals,
    bool known_filled
) :  // variable definitions
    n_rows(dist.n_rows),
    n_cols(dist.n_cols),
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
    this->known_filled = known_filled;
};

template <typename T, typename RNG, SignedInteger sint_t>
SparseSkOp<T,RNG,sint_t>::~SparseSkOp() {
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
template <typename T, typename RNG, SignedInteger sint_t>
RNGState<RNG> fill_sparse(
    SparseSkOp<T,RNG,sint_t> & S
) {
    int64_t long_ax_len = MAX(S.dist.n_rows, S.dist.n_cols);
    int64_t short_ax_len = MIN(S.dist.n_rows, S.dist.n_cols);

    bool is_wide = S.dist.n_rows == short_ax_len;
    sint_t *short_ax_idxs = (is_wide) ? S.rows : S.cols;
    sint_t *long_ax_idxs = (is_wide) ? S.cols : S.rows;

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
    S.known_filled = true;
    return S.next_state;
}

template <typename SKOP>
void print_sparse(SKOP const &S0) {
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
template <typename T, typename RNG, SignedInteger sint_t>
static auto repeated_fisher_yates(
    const RNGState<RNG> &state,
    int64_t vec_nnz,
    int64_t dim_major,
    int64_t dim_minor,
    sint_t *idxs_major,
    sint_t *idxs_minor,
    T *vals
) {
    randblas_error_if(vec_nnz > dim_major);
    std::vector<sint_t> vec_work(dim_major);
    for (sint_t j = 0; j < dim_major; ++j)
        vec_work[j] = j;
    std::vector<sint_t> pivots(vec_nnz);
    RNG gen;
    auto [ctr, key] = state;
    for (sint_t i = 0; i < dim_minor; ++i) {
        sint_t offset = i * vec_nnz;
        auto ctri = ctr;
        ctri.incr(offset);
        for (sint_t j = 0; j < vec_nnz; ++j) {
            // one step of Fisher-Yates shuffling
            auto rv = gen(ctri, key);
            sint_t ell = j + rv[0] % (dim_major - j);
            pivots[j] = ell;
            sint_t swap = vec_work[ell];
            vec_work[ell] = vec_work[j];
            vec_work[j] = swap;
            // update (rows, cols, vals)
            idxs_major[j + offset] = (sint_t) swap;
            vals[j + offset] = (rv[1] % 2 == 0) ? 1.0 : -1.0;
            idxs_minor[j + offset] = (sint_t) i;
            // increment counter
            ctri.incr();
        }
        // Restore vec_work for next iteration of Fisher-Yates.
        //      This isn't necessary from a statistical perspective,
        //      but it makes it easier to generate submatrices of
        //      a given SparseSkOp.
        for (sint_t j = 1; j <= vec_nnz; ++j) {
            sint_t jj = vec_nnz - j;
            sint_t swap = idxs_major[jj + offset];
            sint_t ell = pivots[jj];
            vec_work[jj] = vec_work[ell];
            vec_work[ell] = swap;
        }
        ctr = ctri;
    }
    return RNGState<RNG> {ctr, key};
}

} // end namespace RandBLAS

namespace RandBLAS::sparse {

using RandBLAS::SparseSkOp;
using RandBLAS::MajorAxis;
using RandBLAS::sparse_data::COOMatrix;

template <typename SKOP>
static bool has_fixed_nnz_per_col(
    SKOP const &S0
) {
    if (S0.dist.major_axis == MajorAxis::Short) {
        return S0.dist.n_rows < S0.dist.n_cols;
    } else {
        return S0.dist.n_cols < S0.dist.n_rows;
    }
}

template <typename SKOP>
static int64_t nnz(
    SKOP const &S0
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


template <typename T, typename RNG, SignedInteger sint_t>
COOMatrix<T,sint_t> coo_view_of_skop(SparseSkOp<T,RNG,sint_t> &S) {
    if (!S.known_filled)
        fill_sparse(S);
    int64_t nnz = RandBLAS::sparse::nnz(S);
    COOMatrix<T,sint_t> A(
        S.dist.n_rows, S.dist.n_cols, nnz,
        S.vals, S.rows, S.cols
    );
    return A;
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
static auto transpose(SKOP const &S) {
    randblas_require(S.known_filled);
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

} // end namespace RandBLAS::sparse

#endif
