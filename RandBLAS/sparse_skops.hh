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
#include <unordered_map>

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace RandBLAS::sparse {


template <typename T, SignedInteger sint_t, typename state_t = RNGState<DefaultRNG>>
static state_t repeated_fisher_yates(
    const state_t &state,
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
    using RNG = typename state_t::generator;
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
    ctr.incr(dim_minor * vec_nnz);
    return state_t {ctr, key};
}

inline double isometry_scale(Axis major_axis, int64_t vec_nnz, int64_t dim_major, int64_t dim_minor) {
    if (major_axis == Axis::Short) {
        return std::pow(vec_nnz, -0.5); 
    } else {
        return std::sqrt( ((double) dim_major) / (vec_nnz * ((double) dim_minor)) );
    }
}

}

namespace RandBLAS {

// Forward declaration of SparseSkOp. It's returnable by
// SparseDist.sample(), but its definition involves DenseDist.
template<typename T, typename RNG, SignedInteger sint_t>
struct SparseSkOp;

// =============================================================================
/// A distribution over matrices with structured sparsity. Depending on parameter
/// choices, one can obtain distributions described in the literature as 
/// SJLTs, OSNAPs, hashing embeddings, CountSketch, row or column sampling, or 
/// LESS-Uniform distributions. All members of a SparseDist are const.
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
    ///  Operators sampled from this distribution are constructed by taking independent
    ///  samples from a suitable distribution \math{\mathcal{V}} over sparse vectors.
    ///  This distribution is always over \math{\mathbb{R}^k,}
    ///  where \math{k = \ttt{dim_major}.}  
    ///  The structural properties of \math{\mathcal{V}} depend heavily on whether we're
    ///  short-axis major or long-axis major.
    ///
    ///  To be explicit, let's say that \math{\mtxx} is a sample from \math{\mathcal{V}.}
    ///  
    ///  If \math{\ttt{major_axis} = \ttt{Short}}, then \math{\mtxx} has exactly \math{\vecnnz} nonzeros,
    ///  and the locations of those nonzeros are chosen uniformly
    ///  without replacement from \math{\\{0,\ldots,k-1\\}.} The values of the nonzeros are
    ///  sampled independently and uniformly from +/- 1.
    ///
    ///  If \math{\ttt{major_axis} = \ttt{Long}}, then \math{\mtxx} has *at most* \math{\vecnnz} nonzero
    ///  entries. The locations of the nonzeros are determined by sampling uniformly
    ///  with replacement from \math{\\{0,\ldots,k-1\\}.}
    ///  If index \math{j} occurs in the sample \math{\ell} times, then 
    ///  \math{\mtxx_j} will equal \math{\sqrt{\ell}} with probability 1/2 and
    ///  \math{-\sqrt{\ell}} with probability 1/2.
    ///
    const Axis major_axis;

    // ---------------------------------------------------------------------------
    ///  Defined as
    ///  @verbatim embed:rst:leading-slashes
    ///
    ///  .. math::
    ///
    ///      \ttt{dim_major} = \begin{cases} \,\min\{ \ttt{n_rows},\, \ttt{n_cols} \} &\text{ if }~~ \ttt{major_axis} = \ttt{Short} \\ \max\{ \ttt{n_rows},\,\ttt{n_cols} \} & \text{ if } ~~\ttt{major_axis} = \ttt{Long} \end{cases}.
    ///
    ///  @endverbatim
    const int64_t dim_major;

    // ---------------------------------------------------------------------------
    ///  Defined as \math{\ttt{n_rows} + \ttt{n_cols} - \ttt{dim_major}.} This is
    ///  just whichever of \math{(\ttt{n_rows},\, \ttt{n_cols})} wasn't identified
    ///  as \math{\ttt{dim_major}.}
    const int64_t dim_minor;

    // ---------------------------------------------------------------------------
    ///  An operator sampled from this distribution should be multiplied
    ///  by this constant in order for sketching to preserve norms in expectation.
    const double isometry_scale;

    // ---------------------------------------------------------------------------
    /// This constrains the number of nonzeros in each major-axis vector.
    /// It's subject to the bounds \math{1 \leq \vecnnz \leq \ttt{dim_major}.}
    /// See @verbatim embed:rst:inline :ref:`this tutorial page <sparsedist_params>` for advice on how to set this member. @endverbatim 
    const int64_t vec_nnz;

    // ---------------------------------------------------------------------------
    ///  An upper bound on the number of structural nonzeros that can appear in an
    ///  operator sampled from this distribution. Computed automatically as
    ///  \math{\ttt{full_nnz} = \vecnnz * \ttt{dim_minor}.}
    const int64_t full_nnz;

    // ---------------------------------------------------------------------------
    ///  Arguments passed to this function are used to initialize members of the same names.
    ///  The members \math{\ttt{dim_major},} \math{\ttt{dim_minor},} \math{\ttt{isometry_scale},} and \math{\ttt{full_nnz}}
    ///  are automatically initialized to be consistent with these arguments.
    ///  
    ///  This constructor will raise an error if \math{\min\\{\ttt{n_rows}, \ttt{n_cols}\\} \leq 0} or if 
    ///  \math{\vecnnz} does not respect the bounds documented for the \math{\vecnnz} member.
    SparseDist(
        int64_t n_rows,
        int64_t n_cols,
        int64_t vec_nnz = 4,
        Axis major_axis = Axis::Short
    ) : n_rows(n_rows), n_cols(n_cols),
        major_axis(major_axis),
        dim_major((major_axis == Axis::Short) ? std::min(n_rows, n_cols) : std::max(n_rows, n_cols)),
        dim_minor(n_rows + n_cols - dim_major),
        isometry_scale(sparse::isometry_scale(major_axis, vec_nnz, dim_major, dim_minor)),
        vec_nnz(vec_nnz), full_nnz(vec_nnz * dim_minor) 
    {   // argument validation
        randblas_require(n_rows > 0);
        randblas_require(n_cols > 0);
        randblas_require(vec_nnz > 0);
        randblas_require(vec_nnz <= dim_major);
    }

    // -------------------------------------------------------------------------------------
    ///  Construct a SparseSkOp with this distribution and the provided seed_state.
    template <typename T, typename RNG = DefaultRNG, SignedInteger sint_t = int64_t>
    SparseSkOp<T,RNG,sint_t> sample(RNGState<RNG> &seed_state) {
        return {*this, seed_state};
    }


    // A convenience constructor designed to gracefully handle the common case when someone specifies
    // the short-axis-vector length as a floating point multiple of some other integer. We cast both
    // dimensions to int64_t and raise a warning if that cast is lossy.
    //
    // This function is not part of the public API.
    template <typename ordinal_t1, typename ordinal_t2>
    SparseDist(
        ordinal_t1 n_rows,
        ordinal_t2 n_cols,
        int64_t vec_nnz = 4,
        Axis major_axis = Axis::Short
    ) : SparseDist(cast_int64t(n_rows), cast_int64t(n_cols), vec_nnz, major_axis) { }
};


// =============================================================================
/// This function is used for sampling a sequence of \math{k} elements uniformly
/// without replacement from the index set \math{\\{0,\ldots,n-1\\}.} It uses a special 
/// implementation of Fisher-Yates shuffling to produce \math{r} such samples in \math{O(n + rk)} time.
/// These samples are stored by  writing to \math{\ttt{samples}} in \math{r} blocks of length \math{k.}
/// 
/// The returned RNGState should
/// be used for the next call to a random sampling function whose output should be statistically
/// independent from \math{\ttt{samples}.}
///
template <SignedInteger sint_t, typename state_t = RNGState<DefaultRNG>>
inline state_t repeated_fisher_yates(
    int64_t k, int64_t n, int64_t r, sint_t *samples, const state_t &state
) {
    return sparse::repeated_fisher_yates(state, k, n, r, samples, (sint_t*) nullptr, (double*) nullptr);
}

template <typename RNG = DefaultRNG>
RNGState<RNG> compute_next_state(SparseDist dist, RNGState<RNG> state) {
    int64_t num_mavec, incrs_per_mavec;
    if (dist.major_axis == Axis::Short) {
        num_mavec = std::max(dist.n_rows, dist.n_cols);
        incrs_per_mavec = dist.vec_nnz;
        // ^ SASOs don't try to be frugal with CBRNG increments.
        //   See repeated_fisher_yates.
    } else {
        num_mavec = std::min(dist.n_rows, dist.n_cols);
        incrs_per_mavec = (int64_t) std::ceil((double) dist.vec_nnz / ((double) state.len_c/2));
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
template <typename T, typename RNG = DefaultRNG, SignedInteger sint_t = int64_t>
struct SparseSkOp {

    // ---------------------------------------------------------------------------
    /// Type alias.
    using distribution_t = SparseDist;

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

    /// ---------------------------------------------------------------------------
    ///  **Standard constructor**. Arguments passed to this function are 
    ///  used to initialize members of the same names. own_memory is initialized to true,
    ///  nnz is initialized to -1, and (vals, rows, cols) are each initialized
    ///  to nullptr. next_state is computed automatically from dist and seed_state.
    ///  
    ///  Although own_memory is initialized to true, RandBLAS will not attach
    ///  memory to (vals, rows, cols) unless fill_sparse(SparseSkOp &S) is called. 
    ///
    ///  If a RandBLAS function needs an explicit representation of this operator and
    ///  yet nnz < 0, then RandBLAS will construct a temporary
    ///  explicit representation of this operator and delete that representation before returning.
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
    ///  used to initialize members of the same names. own_memory is initialized to false.
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
/// Note: the "nosub" suffix in this function name reflects how it has no ability
/// to sample submatrices of sparse sketching operators. A future release of
/// RandBLAS will add a function called "fill_sparse_unpacked()" with capabilities
/// that are completely analogous to fill_dense_unpacked().
/// 
template <typename T, typename sint_t, typename state_t>
state_t fill_sparse_unpacked_nosub(
    const SparseDist &D,
    int64_t &nnz, T* vals, sint_t* rows, sint_t *cols,
    const state_t &seed_state
) {
    int64_t dim_major = D.dim_major;
    int64_t dim_minor = D.dim_minor;

    sint_t *idxs_short = (D.n_rows <= D.n_cols) ? rows : cols;
    sint_t *idxs_long  = (D.n_rows <= D.n_cols) ? cols : rows;
    int64_t vec_nnz  = D.vec_nnz;

    if (D.major_axis == Axis::Short) {
        auto state = sparse::repeated_fisher_yates(
            seed_state, vec_nnz, dim_major, dim_minor, idxs_short, idxs_long, vals
        );
        nnz = vec_nnz * dim_minor;
        return state;
    } else {
        // We're long-axis major.
        
        // We don't sample all at once since we might need to merge duplicate entries
        // in each long-axis vector. The way we do this is different than the
        // standard COOMatrix convention of just adding entries together.

        // We begin by defining some datastructures that we repeatedly pass to a helper function.
        // See the comments in the helper function for info on what these guys mean.
        std::unordered_map<sint_t, T> loc2count{};
        std::unordered_map<sint_t, T> loc2scale{}; 
        int64_t total_nnz = 0;
        auto state = seed_state;
        for (int64_t i = 0; i < dim_minor; ++i) {
            state = sample_indices_iid_uniform(dim_major, vec_nnz, idxs_long, vals, state);
            // ^ That writes directly so S.vals and either S.rows or S.cols.
            //   The new values might need to be changed if there are duplicates in idxs_long.
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
    }
}


// =============================================================================
/// If \math{\ttt{S.own_memory}} is true then we enter an allocation stage. This stage
/// inspects the reference members of \math{\ttt{S}}.
/// Any reference member that's equal to \math{\ttt{nullptr}} is redirected to 
/// the start of a new array (allocated with ``new []``) of length \math{\ttt{S.dist.full_nnz}.} 
///
/// After the allocation stage, we inspect the reference members of \math{\ttt{S}}
/// and we raise an error if any of them are null.
///
/// If all reference members are are non-null, then we'll assume each of them has length 
/// at least \math{\ttt{S.dist.full_nnz}.} We'll proceed to populate those members 
/// (and \math{\ttt{S.nnz}}) with the data for the explicit representation of \math{\ttt{S}.}
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
    fill_sparse_unpacked_nosub(S.dist, S.nnz, S.vals, S.rows, S.cols, S.seed_state);
    // ^ We ignore the return value from that function call.
    return;
}

#ifdef __cpp_concepts
static_assert(SketchingDistribution<SparseDist>);
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
        std::cout << "\trows is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (S0.cols != nullptr) {
        std::cout << "\tvector of column indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << S0.cols[i] << ", ";
        }
    } else {
        std::cout << "\tcols is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (S0.vals != nullptr) {
        std::cout << "\tvector of values\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << S0.vals[i] << ", ";
        }
    } else {
        std::cout << "\tvals is the null pointer.\n\t\t";
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
COOMatrix<T, sint_t> coo_view_of_skop(const SparseSkOp &S) {
    randblas_require(S.nnz > 0);
    COOMatrix<T, sint_t> A(S.n_rows, S.n_cols, S.nnz, S.vals, S.rows, S.cols);
    return A;
}


} // end namespace RandBLAS::sparse
