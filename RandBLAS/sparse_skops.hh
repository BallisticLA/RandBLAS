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
    ///  Matrices drawn from this distribution have this many rows;
    ///  must be greater than zero.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many columns;
    ///  must be greater than zero.
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
    /// This sets the number of structural nonzeros in each major-axis vector.
    /// It's subject to the bounds
    /// @verbatim embed:rst:leading-slashes
    ///
    /// .. math::
    ///
    ///     1 \leq \ttt{vec_nnz} \leq \begin{cases} \min\{ \ttt{n_rows}, \ttt{n_cols} \} &\text{ if } \ttt{major_axis} = \ttt{Short} \\ \max\{ \ttt{n_rows},\ttt{n_cols} \} & \text{ if } \ttt{major_axis} = \ttt{Long} \end{cases} 
    ///
    /// @endverbatim
    /// We'll take a moment to provide guidance on how to set this parameter,
    /// without dwelling on formalisms.
    ///
    /// All else equal, larger values of \math{\ttt{vec_nnz}} result in distributions
    /// that are "better" at preserving Euclidean geometry when sketching.
    /// The value of \math{\ttt{vec_nnz}} that suffices for a given context will 
    /// also depend on the sketch size, \math{d := \min\{\ttt{n_rows},\ttt{n_cols}\}},
    /// since larger sketch sizes make it possible to "get away with" smaller values of
    /// \math{\ttt{vec_nnz}}.
    ///
    /// In the short-axis-major case it is usually fine to choose very small values for 
    /// \math{\ttt{vec_nnz}}. For example, suppose we're seeking a constant-distortion embedding
    /// of an unknown subspace of dimension \math{n} where \math{1{,}000 \leq n \leq 10{,}000}.
    /// If we use a short-axis-major sparse distribution \math{d = 2n}, then many practitioners
    /// would feel comfortable taking \math{\ttt{vec_nnz}} as 8 or even 2.
    /// If \math{d} is *much* larger than \math{n} (but still smaller than 
    /// \math{\max\{\ttt{n_rows},\ttt{n_cols}\}}}) then it can be possible to obtain rigorous
    /// gaurantees of sketch quality even when \math{\ttt{vec_nnz}=1};
    /// those familiar with the sketching literature will notice that a short-axis-major 
    /// SparseDist with \math{\ttt{vec_nnz}=1} is usually called a *CountSketch*.
    /// 
    /// We don't have simple recommendations for the long-axis-major case. 
    ///
    const int64_t vec_nnz;

    // ---------------------------------------------------------------------------
    ///  An upper bound on the number of structural nonzeros that can appear in a
    ///  sketching operator sampled from this distribution. This is computed
    ///  automatically as a function of the other members of the SparseDist.
    const int64_t full_nnz;

    // ---------------------------------------------------------------------------
    ///  Constructs a distribution over sparse matrices whose major-axis vectors
    ///  are independent and have exactly vec_nnz nonzeros each. Nonzero entries
    ///  are +/- 1 with equal probability.
    SparseDist(
        int64_t n_rows,
        int64_t n_cols,
        MajorAxis ma,
        int64_t vec_nnz
    ) : n_rows(n_rows), n_cols(n_cols), major_axis(ma),
        isometry_scale(sparse::isometry_scale(ma, vec_nnz, n_rows, n_cols)),
        vec_nnz(vec_nnz),
        full_nnz(sparse::nnz_requirement(ma, vec_nnz, n_rows, n_cols)) 
    {   // argument validation
        randblas_require(n_rows > 0);
        randblas_require(n_cols > 0);
        randblas_require(vec_nnz > 0);
    }
};

template <typename RNG, SignedInteger sint_t>
inline RNGState<RNG> repeated_fisher_yates(
    const RNGState<RNG> &state, int64_t k, int64_t n, int64_t r, sint_t *indices
) {
    return sparse::repeated_fisher_yates(state, k, n, r, indices, (sint_t*) nullptr, (double*) nullptr);
}

template <typename RNG>
RNGState<RNG> compute_next_state(SparseDist dist, RNGState<RNG> state) {
    int64_t num_mavec, incrs_per_mavec;
    if (dist.major_axis == MajorAxis::Short) {
        num_mavec = std::max(dist.n_rows, dist.n_cols);
        incrs_per_mavec = dist.vec_nnz;
        // ^ SASOs don't try to be frugal with CBRNG increments.
        //   See repeated_fisher_yates.
    } else {
        num_mavec = std::min(dist.n_rows, dist.n_cols);
        incrs_per_mavec = dist.vec_nnz * ((int64_t) state.len_c/2);
        // ^ LASOs do try to be frugal with CBRNG increments.
        //   See sample_indices_iid_uniform.
    }
    int64_t full_incr = num_mavec * incrs_per_mavec;
    state.counter.incr(full_incr);
    return state;
}

// =============================================================================
/// A sample from a distribution over structured sparse matrices with either
/// independent rows or independent columns. This type conforms to the
/// SketchingOperator concept.
template <typename T, typename RNG = r123::Philox4x32, SignedInteger sint_t = int64_t>
struct SparseSkOp {

    // ---------------------------------------------------------------------------
    /// Type alias.
    using state_t = RNGState<RNG>;

    // ---------------------------------------------------------------------------
    /// Real scalar type used for nonzeros in matrix representations of this operator.
    using scalar_t = T;

    // ---------------------------------------------------------------------------
    /// Signed integer type used in index arrays for sparse matrix representations
    /// of this operator.
    using index_t = sint_t;

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
    ///  We need workspace to store a representation of the sampled sketching
    ///  operator. This member indicates who is responsible for allocating and 
    ///  deallocating this workspace. If own_memory is true, then 
    ///  RandBLAS is responsible.
    bool own_memory = true; 
    
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    // ---------------------------------------------------------------------------
    ///  The number of structural nonzeros in this operator. If dist.major_axis
    ///  is Short then we know ahead of time that nnz=dist.full_nnz.
    ///  Otherwise, the precise value of nnz can't be known until the operator's
    ///  explicit representation is sampled (although it's always subject to the
    ///  bounds 1 <= nnz <= dist.full_nnz.
    ///
    ///  Negative values are interpreted as a flag that the operator's explicit
    ///  representation hasn't been sampled yet.
    int64_t nnz = -1;

    // ---------------------------------------------------------------------------
    ///  Pointer to values of structural nonzeros in this operator's
    ///  COO-format sparse matrix representation.
    T *vals = nullptr;

    // ---------------------------------------------------------------------------
    ///  Pointer to row indices of structural nonzeros in this operator's
    ///  COO-format sparse matrix representation.
    sint_t *rows = nullptr;

    // ---------------------------------------------------------------------------
    ///  Pointer to column indices of structural nonzeros in this operator's
    ///  COO-format sparse matrix representation.
    sint_t *cols = nullptr;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    ///---------------------------------------------------------------------------
    /// **Memory-owning constructor**. Arguments passed to this function are 
    /// assigned to members of the same name. This constructor allocates
    /// arrays for the \math{\ttt{rows}}, \math{\ttt{cols}}, and \math{\ttt{vals}}
    /// members. Each array has length \math{\ttt{dist.full_nnz}}
    /// and is freed in the destructor unless our reference to it is 
    /// reassigned to nullptr.
    ///
    SparseSkOp(
        SparseDist dist,
        const state_t &seed_state
    ):  // variable definitions
        dist(dist),
        seed_state(seed_state),
        next_state(compute_next_state(dist, seed_state)),
        n_rows(dist.n_rows),
        n_cols(dist.n_cols)
    {   // actual work
        // int64_t nnz = dist.full_nnz;
        // this->rows = new sint_t[nnz]{-1};
        // this->cols = new sint_t[nnz]{-1};
        // this->vals = new T[nnz]{0.0};
    }

    /// --------------------------------------------------------------------------------
    /// **View constructor**. The arguments provided to this
    /// function are used to initialize members of the same names, with no error checking.
    ///
    /// Subsequent calls to RandBLAS functions will assume that 
    /// each of the arrays \math{(\ttt{rows},\ttt{cols},\ttt{vals})} has length at least 
    /// \math{\ttt{dist.full_nnz}}. Additionally,
    /// other RandBLAS functions will ...
    ///
    /// @endverbatim   
    SparseSkOp(
        SparseDist dist,
        const state_t &seed_state,
        const state_t &next_state,
        int64_t nnz,
        T *vals,
        sint_t *rows,
        sint_t *cols
    ) : // variable definitions
        dist(dist),
        seed_state(seed_state),
        next_state(next_state),
        n_rows(dist.n_rows),
        n_cols(dist.n_cols),
        own_memory(false),
        nnz(nnz), vals(vals), rows(rows), cols(cols){ };

    //  Move constructor
    SparseSkOp(SparseSkOp<T,RNG,sint_t> &&S
    ) : dist(S.dist), seed_state(S.seed_state), next_state(S.next_state),
        n_rows(dist.n_rows), n_cols(dist.n_cols), own_memory(S.own_memory),
        rows(S.rows), cols(S.cols), vals(S.vals)
    {
        S.rows = nullptr;
        S.cols = nullptr;
        S.vals = nullptr;
    }

    //  Destructor
    ~SparseSkOp() {
        if (this->own_memory) {
            if (this->rows != nullptr)
                delete [] this->rows;
            if (this->cols != nullptr)
                delete [] this->cols;
            if (this->vals != nullptr)
                delete [] this->vals;
        }
    }
};

static_assert(SketchingDistribution<SparseDist>);

template <typename T, SignedInteger sint_t>
void laso_merge_long_axis_vector_coo_data(
    int64_t vec_nnz, T* vals, sint_t* idxs_lax, sint_t *idxs_sax, int64_t i,
    std::unordered_map<sint_t, T> &loc2count,
    std::unordered_map<sint_t, T> &loc2scale
) {
    loc2count.clear();
    // ^ Used to count the number of times each long-axis index
    //   appears in a given long-axis vector. Indices that don't
    //   appear are not stored explicitly.
    loc2scale.clear();
    // ^ Stores a mean-zero variance-one subgaussian random variable for
    //   each index appearing in the long-axis vector. Current
    //   long-axis-sparse sampling uses Rademachers, but the literature
    //   technically prefers Gaussians.
    for (int64_t j = 0; j < vec_nnz; ++j) {
        idxs_sax[j] = i;
        sint_t ell = idxs_lax[j];
        T      val = vals[j];
        if (loc2scale.count(ell)) {
            loc2count[ell] = loc2count[ell] + 1;
        } else {
            loc2scale[ell] = val;
            loc2count[ell] = 1.0;
        }
    }
    if ((int64_t) loc2scale.size() < vec_nnz) {
        // Then we have duplicates. We need to overwrite some of the values
        // of (idxs_lax, vals, idxs_sax) and implicitly
        // shift them backward to remove duplicates;
        int64_t count = 0;
        for (const auto& [ell,c] : loc2count) {
            idxs_lax[count] = ell;
            vals[count] = std::sqrt(c) * loc2scale[ell];
            count += 1;
        }
    }
    return;
}

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
    using sint_t = typename SparseSkOp::index_t;
    using RNG    = typename SparseSkOp::state_t::generator;
    using T      = typename SparseSkOp::scalar_t;

    int64_t full_nnz = S.dist.full_nnz;
    if (S.own_memory) {
        if (S.rows == nullptr) S.rows = new sint_t[full_nnz];
        if (S.cols == nullptr) S.cols = new sint_t[full_nnz];
        if (S.vals == nullptr) S.vals = new T[full_nnz];
    }
    randblas_require(S.rows != nullptr);
    randblas_require(S.cols != nullptr);
    randblas_require(S.vals != nullptr);

    int64_t dim_long  = MAX(S.dist.n_rows, S.dist.n_cols);
    int64_t dim_short = MIN(S.dist.n_rows, S.dist.n_cols);

    bool is_wide = S.dist.n_rows == dim_short;
    sint_t *idxs_short = (is_wide) ? S.rows : S.cols;
    sint_t *idxs_long  = (is_wide) ? S.cols : S.rows;
    int64_t vec_nnz  = S.dist.vec_nnz;
    T* vals = S.vals;

    if (S.dist.major_axis == MajorAxis::Short) {
        sparse::repeated_fisher_yates(
            S.seed_state, vec_nnz, dim_short, dim_long, idxs_short, idxs_long, vals
        );
        S.nnz = vec_nnz * dim_long;
    } else {
        // We don't sample all at once since we might need to merge duplicate entries
        // in each long-axis vector. The way we do this is different than the
        // standard COOMatrix convention of just adding entries together.

        // We begin by defining some datastructures that we repeatedly pass to a helper function.
        // See the comments in the helper function for info on what these guys mean.
        std::unordered_map<sint_t, T> loc2count{};
        std::unordered_map<sint_t, T> loc2scale{}; 
        int64_t total_nnz = 0;
        auto state = S.seed_state;
        for (int64_t i = 0; i < dim_short; ++i) {
            state = sample_indices_iid_uniform<RNG,T,true>(dim_long, vec_nnz, idxs_long, vals, state);
            // ^ That writes directly so S.vals and either S.rows or S.cols.
            //   The new values might need to be changed if there are duplicates in lind.
            //   We have a helper function for this since it's a tedious process.
            //   The helper function also sets whichever of S.rows or S.cols wasn't populated.
            laso_merge_long_axis_vector_coo_data(
                vec_nnz, vals, idxs_long, idxs_short, i, loc2count, loc2scale
            );
            int64_t count = loc2count.size();
            vals += count;
            idxs_long  += count;
            idxs_short += count;
            total_nnz  += count;
        }
        // We sanitize any space we didn't end up using.
        int64_t len_slack = S.dist.full_nnz - total_nnz;
        for (int64_t i = 0; i < len_slack; ++i) {
            idxs_long[i]  = 0;
            idxs_short[i] = 0;
            vals[i] = 0;
        }
        S.nnz = total_nnz;
    }
    return;
}

template <typename SparseSkOp>
void print_sparse(SparseSkOp const &S0) {
    // TODO: clean up this function.
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
    if (S0.rows != nullptr) {
        std::cout << "\tvector of row indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << S0.rows[i] << ", ";
        }
    } else {
        std::cout << "\tthis->rows is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (S0.cols != nullptr) {
        std::cout << "\tvector of column indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << S0.cols[i] << ", ";
        }
    } else {
        std::cout << "\tthis->cols is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (S0.vals != nullptr) {
        std::cout << "\tvector of values\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << S0.vals[i] << ", ";
        }
    } else {
        std::cout << "\tthis->vals is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    return;
}

} // end namespace RandBLAS

namespace RandBLAS::sparse {

using RandBLAS::SparseSkOp;
using RandBLAS::MajorAxis;
using RandBLAS::sparse_data::COOMatrix;

template <typename SparseSkOp, typename T = SparseSkOp::scalar_t, typename sint_t = SparseSkOp::index_t>
COOMatrix<T, sint_t> coo_view_of_skop(SparseSkOp &S) {
    if (S.nnz <= 0)
        fill_sparse(S);
    COOMatrix<T, sint_t> A(S.n_rows, S.n_cols, S.nnz, S.vals, S.rows, S.cols);
    return A;
}


} // end namespace RandBLAS::sparse
