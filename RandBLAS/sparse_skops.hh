// Copyright, 2024. See LICENSE for copyright holder information.
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

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace RandBLAS::sparse {


// =============================================================================
/// WARNING: this function is not part of the public API.
///
template <typename T, typename RNG, SignedInteger sint_t>
static RNGState<RNG> repeated_fisher_yates(
    const RNGState<RNG> &state,
    int64_t vec_nnz,
    int64_t dim_major,
    int64_t dim_minor,
    sint_t *idxs_major,
    sint_t *idxs_minor,
    T *vals
) {
    bool write_vals = vals != nullptr;
    bool write_idxs_minor = idxs_minor != nullptr;
    randblas_error_if(vec_nnz > dim_major);
    std::vector<sint_t> vec_work(dim_major);
    for (sint_t j = 0; j < dim_major; ++j)
        vec_work[j] = j;
    std::vector<sint_t> pivots(vec_nnz);
    RNG gen;
    auto [ctr, key] = state;
    for (sint_t i = 0; i < dim_minor; ++i) {
        sint_t offset = i * vec_nnz;
        auto ctr_work = ctr;
        ctr_work.incr(offset);
        for (sint_t j = 0; j < vec_nnz; ++j) {
            // one step of Fisher-Yates shuffling
            auto rv = gen(ctr_work, key);
            sint_t ell = j + rv[0] % (dim_major - j);
            pivots[j] = ell;
            sint_t swap = vec_work[ell];
            vec_work[ell] = vec_work[j];
            vec_work[j] = swap;
            // update (rows, cols, vals)
            idxs_major[j + offset] = (sint_t) swap;
            if (write_vals)
                vals[j + offset] = (rv[1] % 2 == 0) ? 1.0 : -1.0;
            if (write_idxs_minor)
                idxs_minor[j + offset] = (sint_t) i;
            // increment counter
            ctr_work.incr();
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
    }
    return RNGState<RNG> {ctr, key};
}

inline double isometry_scale(MajorAxis ma, int64_t vec_nnz, int64_t n_rows, int64_t n_cols) {
    if (ma == MajorAxis::Short) {
        return std::pow(vec_nnz, -0.5); 
    } else if (ma == MajorAxis::Long) {
        double minor_ax_len = std::min(n_rows, n_cols);
        double major_ax_len = std::max(n_rows, n_cols);
        return std::sqrt( major_ax_len / (vec_nnz * minor_ax_len) );
    } else {
        throw std::invalid_argument("Cannot compute the isometry scale for a sparse operator with unspecified major axis.");
    }
}

static int64_t nnz_requirement(MajorAxis ma, int64_t vec_nnz, int64_t n_rows, int64_t n_cols) {
    bool saso = ma == MajorAxis::Short;
    bool wide = n_rows < n_cols;
    if (saso & wide) {
        return vec_nnz * n_cols;
    } else if (saso & (!wide)) {
        return vec_nnz * n_rows;
    } else if (wide & (!saso)) {
        return vec_nnz * n_rows;
    } else {
        // tall LASO
        return vec_nnz * n_cols;
    }
}

}

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
    ///  Constrains the sparsity pattern of matrices drawn from this distribution. 
    ///
    ///  Having major_axis==Short results in sketches are more likely to contain
    ///  useful geometric information, without making assumptions about the data
    ///  being sketched.
    ///
    const MajorAxis major_axis = MajorAxis::Short;

    // ---------------------------------------------------------------------------
    ///  A sketching operator sampled from this distribution should be multiplied
    ///  by this constant in order for sketching to preserve norms in expectation.
    const double isometry_scale;

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
    ///  An upper bound on the number of structural nonzeros that can appear in a
    ///  sketching operator sampled from this distribution. This is computed
    ///  automatically as a function of the other members of the SparseDist.
    const int64_t full_nnz;


    SparseDist(
        int64_t n_rows,
        int64_t n_cols,
        MajorAxis ma,
        int64_t vec_nnz
    ) : n_rows(n_rows), n_cols(n_cols), major_axis(ma),
        isometry_scale(sparse::isometry_scale(ma, vec_nnz, n_rows, n_cols)),
        vec_nnz(vec_nnz),
        full_nnz(sparse::nnz_requirement(ma, vec_nnz, n_rows, n_cols)) {}
};

template <typename RNG, SignedInteger sint_t>
inline RNGState<RNG> repeated_fisher_yates(
    const RNGState<RNG> &state, int64_t k, int64_t n, int64_t r, sint_t *indices
) {
    return sparse::repeated_fisher_yates(state, k, n, r, indices, (sint_t*) nullptr, (double*) nullptr);
}

template <typename RNG>
RNGState<RNG> compute_next_state(SparseDist dist, RNGState<RNG> state) {
    int64_t minor_len;
    if (dist.major_axis == MajorAxis::Short) {
        minor_len = std::min(dist.n_rows, dist.n_cols);
    } else {
        minor_len = std::max(dist.n_rows, dist.n_cols);
    }
    int64_t full_incr = minor_len * dist.vec_nnz;
    state.counter.incr(full_incr);
    return state;
}

// =============================================================================
/// Represents a sample from a distribution over structured sparse matrices.
/// 
/// This type conforms to the SketchingOperator concept.
template <typename T, typename RNG = r123::Philox4x32, SignedInteger sint_t = int64_t>
struct SparseSkOp {

    // ---------------------------------------------------------------------------
    /// Signed integer type used in index arrays for sparse matrix representations
    /// of this operator.
    using index_t = sint_t;

    // ---------------------------------------------------------------------------
    /// Type alias.
    using state_t = RNGState<RNG>;

    // ---------------------------------------------------------------------------
    /// Real scalar type used for nonzeros in matrix representations of this operator.
    using scalar_t = T;

    // ---------------------------------------------------------------------------
    ///  The distribution from which this sketching operator is sampled.
    ///  This member specifies the number of rows and columns of the sketching
    ///  operator. It also controls the sparsity pattern and the sparsity level.
    const SparseDist dist;

    // ---------------------------------------------------------------------------
    ///  The state that should be passed to the RNG when the full sketching 
    ///  operator needs to be sampled from scratch. 
    const state_t seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used by the next call to an RNG *after* the
    ///  full sketching operator has been sampled.
    const state_t next_state;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_cols.
    const int64_t n_cols;

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

    ///---------------------------------------------------------------------------
    /// Memory-owning constructor for a SparseSkOp. This will internally allocate
    /// three arrays of length dist.full_nnz. The arrays will be freed when this
    /// object's destructor is called.
    ///
    SparseSkOp(
        SparseDist dist,
        const state_t &state
    ) :  // variable definitions
        dist(dist),
        seed_state(state),
        next_state(compute_next_state(dist, seed_state)),
        n_rows(dist.n_rows),
        n_cols(dist.n_cols),
        own_memory(true)
    {   // sanity checks
        randblas_require(this->dist.n_rows > 0);
        randblas_require(this->dist.n_cols > 0);
        randblas_require(this->dist.vec_nnz > 0);
        // actual work
        int64_t nnz = dist.full_nnz;
        this->rows = new sint_t[nnz];
        this->cols = new sint_t[nnz];
        this->vals = new T[nnz];
    }

    /// ---------------------------------------------------------------------------
    /// Non-memory-owning constructor for a SparseSkOp. Each of the arrays (rows, cols, vals)
    /// must have length at least dist.full_nnz. We assume these arrays already contain the
    /// defining data for a COO-representation of this operator as a sparse matrix. If you 
    /// want to override this behavior and only want to treat (rows, cols, vals) as workspace,
    /// set the final argument of this function to known_filled=false.
    ///
    /// @verbatim embed:rst:leading-slashes
    /// .. dropdown:: Full parmaeter descriptions
    ///   :animate: fade-in-slide-down
    ///
    ///     dist
    ///      - A SparseDist object.
    ///      - Defines the number of rows and columns in this sketching operator.
    ///      - Indirectly controls sparsity pattern, and directly controls sparsity level.
    /// 
    ///     state
    ///      - An RNGState object.
    ///      - The RNG will use this as the starting point to generate all 
    ///        random numbers needed for this sketching operator. Requred
    ///        even if known_filled=True.
    /// 
    ///     rows
    ///      - Pointer to array array of of signed integers (type sint_t, defaults to int64_t).
    ///      - stores row indices as part of the COO format.
    /// 
    ///     cols
    ///      - Pointer to array array of of signed integers (type sint_t, defaults to int64_t).
    ///      - stores column indices as part of the COO format.
    /// 
    ///     vals
    ///      - Pointer to array of real numerical type T.
    ///      - stores structural nonzeros as part of the COO format.
    ///  
    ///     known_filled
    ///      - A boolean, which defaults to true.
    ///      - If true, then the arrays pointed to by
    ///        (rows, cols, vals) already contain the randomly sampled
    ///        data defining this sketching operator.
    ///
    /// @endverbatim   
    SparseSkOp(
        SparseDist dist,
        const state_t &state,
        sint_t *rows,
        sint_t *cols,
        T *vals,
        bool known_filled = true
    ) : // variable definitions
        dist(dist),
        seed_state(state),
        n_rows(dist.n_rows),
        n_cols(dist.n_cols),
        own_memory(false),
        next_state(compute_next_state(dist, seed_state))
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

    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        sint_t *rows,
        sint_t *cols,
        T *vals 
    ) : SparseSkOp(dist, RNGState<RNG>(key), rows, cols, vals) {};

    SparseSkOp(
        SparseDist dist,
        uint32_t key
    ) : SparseSkOp(dist, RNGState<RNG>(key)) {};


    //  Destructor
    ~SparseSkOp() {
        if (this->own_memory) {
            delete [] this->rows;
            delete [] this->cols;
            delete [] this->vals;
        }
    }
};

static_assert(SketchingDistribution<SparseDist>);

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
template <typename SparseSkOp>
void fill_sparse(SparseSkOp &S) {
    
    int64_t long_ax_len = MAX(S.dist.n_rows, S.dist.n_cols);
    int64_t short_ax_len = MIN(S.dist.n_rows, S.dist.n_cols);
    bool is_wide = S.dist.n_rows == short_ax_len;

    using sint_t = typename SparseSkOp::index_t;
    sint_t *short_ax_idxs = (is_wide) ? S.rows : S.cols;
    sint_t *long_ax_idxs  = (is_wide) ? S.cols : S.rows;

    if (S.dist.major_axis == MajorAxis::Short) {
        sparse::repeated_fisher_yates(
            S.seed_state, S.dist.vec_nnz, short_ax_len, long_ax_len,
            short_ax_idxs, long_ax_idxs, S.vals
        );
    } else {
        sparse::repeated_fisher_yates(
            S.seed_state, S.dist.vec_nnz, long_ax_len, short_ax_len,
            long_ax_idxs, short_ax_idxs, S.vals
        );
    }
    S.known_filled = true;
    return;
}

template <typename SparseSkOp>
void print_sparse(SparseSkOp const &S0) {
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


} // end namespace RandBLAS

namespace RandBLAS::sparse {

using RandBLAS::SparseSkOp;
using RandBLAS::MajorAxis;
using RandBLAS::sparse_data::COOMatrix;

template <typename SparseSkOp>
static bool has_fixed_nnz_per_col(
    SparseSkOp const &S0
) {
    if (S0.dist.major_axis == MajorAxis::Short) {
        return S0.dist.n_rows < S0.dist.n_cols;
    } else {
        return S0.dist.n_cols < S0.dist.n_rows;
    }
}

template <typename SparseSkOp, typename T = SparseSkOp::scalar_t, typename sint_t = SparseSkOp::index_t>
COOMatrix<T, sint_t> coo_view_of_skop(SparseSkOp &S) {
    if (!S.known_filled)
        fill_sparse(S);
    int64_t nnz = S.dist.full_nnz;
    COOMatrix<T, sint_t> A(S.dist.n_rows, S.dist.n_cols, nnz, S.vals, S.rows, S.cols);
    return A;
}


} // end namespace RandBLAS::sparse
