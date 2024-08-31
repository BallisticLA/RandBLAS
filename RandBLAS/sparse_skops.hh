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

inline double isometry_scale(Axis major_axis, int64_t vec_nnz, int64_t n_rows, int64_t n_cols) {
    if (major_axis == Axis::Short) {
        return std::pow(vec_nnz, -0.5); 
    } else if (major_axis == Axis::Long) {
        double minor_ax_len = std::min(n_rows, n_cols);
        double major_ax_len = std::max(n_rows, n_cols);
        return std::sqrt( major_ax_len / (vec_nnz * minor_ax_len) );
    } else {
        throw std::invalid_argument("Cannot compute the isometry scale for a sparse operator with unspecified major axis.");
    }
}

static int64_t nnz_requirement(Axis major_axis, int64_t vec_nnz, int64_t n_rows, int64_t n_cols) {
    if (major_axis == Axis::Undefined) {
        throw std::invalid_argument("Cannot compute the nnz requirement for a sparse operator with unspecified major axis.");
    }
    bool saso = major_axis == Axis::Short;
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
/// A distribution over matrices with structured sparsity. Depending on parameter
/// choices, one can obtain distributions described in the literature as 
/// SJLTs, OSNAPs, hashing embeddings, CountSketch, row or column sampling, or 
/// LESS-Uniform distributions.
/// 
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
    ///  @verbatim embed:rst:leading-slashes
    ///  Constrains the sparsity pattern of matrices drawn from this distribution. 
    ///
    ///  Several well-known distributions can be recovered by appropriate choices
    ///  of major_axis and vec_nnz.
    ///
    ///  If major_axis == Short:
    ///
    ///     vec_nnz = 1 corresponds to the distribution over CountSketch operators.
    ///     vec_nnz > 1 corresponds to distributions which have been studied under
    ///     many different names, including OSNAPs, SJLTs, and Hashing embeddings.
    ///
    ///  If major_axis == Long:
    ///
    ///     vec_nnz = 1 corresponds to operators for sampling uniformly with replacement
    ///     from the rows or columns of a data matrix (although the signs on the rows or
    ///     columns may be flipped). vec_nnz > 1 corresponds to so-called LESS-uniform
    ///     distributions.
    ///
    ///  For the same value of vec_nnz, short-axis-sparse sketching has (far) better
    ///  statistical properties than long-axis-sparse sketching. However, 
    ///  an operator that's long-axis-sparse with a given value of vec_nnz is 
    ///  far sparser than the corresponding short-axis-sparse operator with the 
    ///  same value for vec_nnz. 
    ///  @endverbatim
    const Axis major_axis;

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
    ///     1 \leq \ttt{vec_nnz} \leq \begin{cases} \min\{ \ttt{n_rows},\, \ttt{n_cols} \} &\text{ if }~~ \ttt{major_axis} = \ttt{Short} \\ \max\{ \ttt{n_rows},\,\ttt{n_cols} \} & \text{ if } ~~\ttt{major_axis} = \ttt{Long} \end{cases} 
    ///
    /// @endverbatim
    /// We'll take a moment to provide guidance on how to set this parameter,
    /// without dwelling on formalisms.
    ///
    /// All else equal, larger values of \math{\ttt{vec_nnz}} result in distributions
    /// that are "better" at preserving Euclidean geometry when sketching.
    /// The value of \math{\ttt{vec_nnz}} that suffices for a given context will 
    /// also depend on the sketch size, \math{d := \min\\{\ttt{n_rows},\ttt{n_cols}\\}.}
    /// Larger sketch sizes make it possible to "get away with" smaller values of
    /// \math{\ttt{vec_nnz}}.
    ///
    /// For short-axis-major sparse sketching fine to choose very small values for 
    /// \math{\ttt{vec_nnz}}. For example, suppose we're seeking a constant-distortion embedding
    /// of an unknown subspace of dimension \math{n} where \math{1{,}000 \leq n \leq 10{,}000}.
    /// If we use a short-axis-major sparse distribution \math{d = 2n}, then many practitioners
    /// would feel comfortable taking \math{\ttt{vec_nnz}} as 8 or even 2.
    /// 
    /// If one seeks similar statistical properties from long-axis-sparse sketching it is
    /// important to use (much) larger values of \math{\ttt{vec_nnz}}. There is less consensus
    /// in the community for what constitutes "big enough in practice," therefore we make
    /// no prescriptions here.
    ///
    const int64_t vec_nnz;

    // ---------------------------------------------------------------------------
    ///  An upper bound on the number of structural nonzeros that can appear in a
    ///  sketching operator sampled from this distribution. This is computed
    ///  automatically as a function of the other members of the SparseDist.
    const int64_t full_nnz;

    // ---------------------------------------------------------------------------
    ///  Arguments passed to this function are used to initialize members of the same names.
    ///  isometry_scale and full_nnz are automatically initialized to values that are
    ///  consistent with these arguments.
    ///  
    ///  This constructor will raise an error if min(n_rows, n_cols) <= 0 or if 
    ///  vec_nnz does not respect the bounds documented for the vec_nnz member.
    SparseDist(
        int64_t n_rows,
        int64_t n_cols,
        Axis major_axis,
        int64_t vec_nnz
    ) : n_rows(n_rows), n_cols(n_cols), major_axis(major_axis),
        isometry_scale(sparse::isometry_scale(major_axis, vec_nnz, n_rows, n_cols)),
        vec_nnz(vec_nnz),
        full_nnz(sparse::nnz_requirement(major_axis, vec_nnz, n_rows, n_cols)) 
    {   // argument validation
        randblas_require(n_rows > 0);
        randblas_require(n_cols > 0);
        randblas_require(vec_nnz > 0);
        if (major_axis == Axis::Short) {
            randblas_require(vec_nnz <= std::min(n_rows, n_cols));
        } else {
            // The initializers will have errored if major_axis == Axis::Undefined,
            // so we can assume that major_axis == Axis::Long.
            randblas_require(vec_nnz <= std::max(n_rows, n_cols));
        }
        
    }
};

#ifdef __cpp_concepts
static_assert(SketchingDistribution<SparseDist>);
#endif

template <typename RNG, SignedInteger sint_t>
inline RNGState<RNG> repeated_fisher_yates(
    const RNGState<RNG> &state, int64_t k, int64_t n, int64_t r, sint_t *indices
) {
    return sparse::repeated_fisher_yates(state, k, n, r, indices, (sint_t*) nullptr, (double*) nullptr);
}

template <typename RNG>
RNGState<RNG> compute_next_state(SparseDist dist, RNGState<RNG> state) {
    int64_t num_mavec, incrs_per_mavec;
    if (dist.major_axis == Axis::Short) {
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
    ///  The distribution from which this operator is sampled.
    const SparseDist dist;

    // ---------------------------------------------------------------------------
    ///  The state passed to random sampling functions when the full
    ///  operator needs to be sampled from scratch. 
    const state_t seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used in the next call to a random sampling function
    ///  whose output should be statistically independent from properties of this
    ///  operator.
    const state_t next_state;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_rows. Automatically initialized in all constructors.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_cols. Automatically initialized in all constructors.
    const int64_t n_cols;

    // ----------------------------------------------------------------------------
    ///  If true, then RandBLAS has permission to allocate and attach memory to this operator's reference
    ///  members (S.rows, S.cols, and S.vals). If true *at destruction time*, then delete []
    ///  will be called on each of this operator's non-null reference members.
    ///
    ///  RandBLAS only writes to this member at construction time.
    ///   
    ///  \internal
    ///  See also: fill_sparse(SparseSkOp &S) and the statement of our 
    ///  @verbatim embed:rst:inline :ref:`memory management policies <memory_tutorial>`. @endverbatim 
    ///  \internal
    bool own_memory;
    
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    // ---------------------------------------------------------------------------
    ///  The number of structural nonzeros in this operator.
    ///  Negative values are a flag that the operator's explicit representation
    ///  hasn't been sampled yet.
    ///
    ///  If this value is negative when this operator is
    ///  passed to sketching function, then that function will
    ///  call fill_sparse(SparseSkOp &S).
    ///
    ///  \internal
    ///  If dist.major_axis
    ///  is Short then we know ahead of time that nnz=dist.full_nnz.
    ///  Otherwise, the precise value of nnz can't be known until the operator's
    ///  explicit representation is sampled (although it's always subject to the
    ///  bounds 1 <= nnz <= dist.full_nnz.
    ///  \endinternal
    ///  
    int64_t nnz;

    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds the values of this operator's structural nonzeros.
    ///
    ///  If non-null, this must point to an array of length at least dist.full_nnz.
    T *vals;

    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds the row indices for this operator's structural nonzeros.
    ///
    ///  If non-null, this must point to an array of length at least dist.full_nnz.
    sint_t *rows;

    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds the column indices for this operator's structural nonzeros.
    ///
    ///  If non-null, this must point to an array of length at least dist.full_nnz.
    sint_t *cols;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    ///---------------------------------------------------------------------------
    ///  **Standard constructor**. Arguments passed to this function are 
    ///  used to initialize members of the same name. own_memory is initialized to true,
    ///  nnz is initialized to -1, and (vals, rows, cols) are each initialized
    ///  to nullptr. next_state is computed automatically from dist and seed_state.
    ///  
    ///  This constructor is intended for use with fill_sparse(SparseSkOp &S),
    ///  which RandBLAS will call if and when needed.
    ///  
    SparseSkOp(
        SparseDist dist,
        const state_t &seed_state
    ):  // variable definitions
        dist(dist),
        seed_state(seed_state),
        next_state(compute_next_state(dist, seed_state)),
        n_rows(dist.n_rows),
        n_cols(dist.n_cols), own_memory(true), nnz(-1), vals(nullptr), rows(nullptr), cols(nullptr) { }

    /// --------------------------------------------------------------------------------
    ///  **Expert constructor**. Arguments passed to this function are 
    ///  used to initialize members of the same name. own_memory is initialized to false.
    /// 
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
        nnz(S.nnz), rows(S.rows), cols(S.cols), vals(S.vals)
    {
        S.rows = nullptr;
        S.cols = nullptr;
        S.vals = nullptr;
        S.nnz = -1;
    }

    //  Destructor
    ~SparseSkOp() {
        if (own_memory) {
            if (rows != nullptr) delete [] rows;
            if (cols != nullptr) delete [] cols;
            if (vals != nullptr) delete [] vals;
        }
    }
};


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
/// @verbatim embed:rst:leading-slashes
///
///   .. |vals|  mathmacro:: \mathtt{vals}
///   .. |rows|  mathmacro:: \mathtt{rows}
///   .. |cols|  mathmacro:: \mathtt{cols}
///   .. |Dfullnnz| mathmacro:: {\mathcal{D}\mathtt{.full\_nnz}}
///
/// @endverbatim
/// This function is the underlying implementation of fill_sparse(SparseSkOp &S).
/// It has no allocation stage and it skips checks for null pointers.
///
/// On entry, \math{(\vals,\rows,\cols)} are arrays of length at least \math{\Dfullnnz.}
/// On exit, the first \math{\ttt{nnz}} entries of these arrays contain the data for 
/// a COO sparse matrix representation of the SparseSkOp
/// defined by \math{(\D,\ttt{seed_state)}.}
///
/// 
template <typename T, typename sint_t, typename state_t>
state_t fill_sparse(
    const SparseDist &D,
    int64_t &nnz, T* vals, sint_t* rows, sint_t *cols,
    const state_t &seed_state
) {
    int64_t dim_long  = MAX(D.n_rows, D.n_cols);
    int64_t dim_short = MIN(D.n_rows, D.n_cols);

    sint_t *idxs_short = (D.n_rows == dim_short) ? rows : cols;
    sint_t *idxs_long  = (D.n_rows == dim_short) ? cols : rows;
    int64_t vec_nnz  = D.vec_nnz;

    if (D.major_axis == Axis::Short) {
        auto state = sparse::repeated_fisher_yates(
            seed_state, vec_nnz, dim_short, dim_long, idxs_short, idxs_long, vals
        );
        nnz = vec_nnz * dim_long;
        return state;
    } else if (D.major_axis == Axis::Long) {
        // We don't sample all at once since we might need to merge duplicate entries
        // in each long-axis vector. The way we do this is different than the
        // standard COOMatrix convention of just adding entries together.

        // We begin by defining some datastructures that we repeatedly pass to a helper function.
        // See the comments in the helper function for info on what these guys mean.
        std::unordered_map<sint_t, T> loc2count{};
        std::unordered_map<sint_t, T> loc2scale{}; 
        int64_t total_nnz = 0;
        auto state = seed_state;
        for (int64_t i = 0; i < dim_short; ++i) {
            using RNG = typename state_t::generator;
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
        nnz = total_nnz;
        return state;
    } else {
        throw std::invalid_argument("D.major_axis must be Axis::Short or Axis::Long.");
    }
}


// =============================================================================
/// If \math{\ttt{S.own_memory}} is true then we enter an allocation stage. This stage
/// inspects the reference members of \math{\ttt{S}}, 
/// and any of them which is equal to \math{\ttt{nullptr}} is redirected to the
/// start of an array allocated with ``new []``. The length of any allocated
/// array is \math{\ttt{S.dist.full_nnz}}. 
///
/// After the allocation stage, we inspect the reference members of \math{\ttt{S}}
/// and we raise an error if any of them are null.
///
/// If all reference members are are non-null, then we'll assume each of them has length 
/// at least \math{\ttt{S.dist.full_nnz}}. We'll proceed to populate those members 
/// (and \math{\ttt{S.nnz}}) with the data for the explicit representation of \math{\ttt{S}}.
/// On exit, \math{\ttt{S}} can be equivalently represented by
/// @verbatim embed:rst:leading-slashes
///  .. code:: c++
///
///         RandBLAS::COOMatrix mat(S.n_rows, S.n_cols, S.nnz, S.vals, S.rows, S.cols);
///
/// @endverbatim
template <typename SparseSkOp>
void fill_sparse(SparseSkOp &S) {
    using sint_t = typename SparseSkOp::index_t;
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
    fill_sparse(S.dist, S.nnz, S.vals, S.rows, S.cols, S.seed_state);
    return;
}

#ifdef __cpp_concepts
static_assert(SketchingOperator<SparseSkOp<float>>);
static_assert(SketchingOperator<SparseSkOp<double>>);
#endif

template <typename SparseSkOp>
void print_sparse(SparseSkOp const &S0) {
    // TODO: clean up this function.
    std::cout << "SparseSkOp information" << std::endl;
    int64_t nnz;
    if (S0.dist.major_axis == Axis::Short) {
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
using RandBLAS::Axis;
using RandBLAS::sparse_data::COOMatrix;

template <typename SparseSkOp, typename T = SparseSkOp::scalar_t, typename sint_t = SparseSkOp::index_t>
COOMatrix<T, sint_t> coo_view_of_skop(SparseSkOp &S) {
    if (S.nnz <= 0)
        fill_sparse(S);
    COOMatrix<T, sint_t> A(S.n_rows, S.n_cols, S.nnz, S.vals, S.rows, S.cols);
    return A;
}


} // end namespace RandBLAS::sparse
