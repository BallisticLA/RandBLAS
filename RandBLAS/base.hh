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

/// @file

#include "RandBLAS/config.h"
#include "RandBLAS/random_gen.hh"

#include <blas.hh>
#include <utility>
#include <cstring>
#include <cstdint>
#include <iostream>

#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#include<iostream>


/// code common across the project
namespace RandBLAS {

typedef r123::Philox4x32 DefaultRNG;
using std::uint64_t;

/// -------------------------------------------------------------------
/// This is a stateful version of a
/// *counter-based random number generator* (CBRNG) from Random123.
/// It packages a CBRNG together with two arrays, called "counter" and "key,"
/// which are interpreted as extended-width unsigned integers.
/// 
/// RNGStates are used in every RandBLAS function that involves random sampling.
///
template <typename RNG = DefaultRNG>
struct RNGState {

    /// -------------------------------------------------------------------
    /// Type of the underlying Random123 CBRNG. Must be based on 
    /// Philox or Threefry. We've found that Philox works best for our 
    /// purposes, and we default to Philox4x32.
    using generator = RNG;
    
    using ctr_type = typename RNG::ctr_type;
    // ^ An array type defined in Random123.
    using key_type = typename RNG::key_type;
    // ^ An array type defined in Random123.
    using ctr_uint = typename RNG::ctr_type::value_type;
    // ^ The unsigned integer type used in this RNGState's counter array.
    using key_uint = typename RNG::key_type::value_type;
    // ^ The unsigned integer type used in this RNGState's key array.

    /// ------------------------------------------------------------------
    /// This is a Random123-defined statically-sized array of unsigned integers.
    /// With RandBLAS' default, it contains four 32-bit unsigned ints
    /// and is interpreted as one 128-bit unsigned int.
    /// 
    /// This member specifies a "location" in the random stream
    /// defined by RNGState::generator and RNGState::key.
    /// Random sampling functions in RandBLAS effectively consume elements
    /// of the random stream starting from this location.
    ///
    /// **RandBLAS functions do not mutate input RNGStates.** Free-functions 
    /// return new RNGStates with suitably updated counters. Constructors
    /// for SketchingOperator objects store updated RNGStates in the
    /// object's next_state member.
    typename RNG::ctr_type counter;

    /// ------------------------------------------------------------------
    /// This is a Random123-defined statically-sized array of unsigned integers.
    /// With RandBLAS' default, it contains two 32-bit unsigned ints
    /// and is interpreted as one 64-bit unsigned int.
    ///
    /// This member specifices a sequece of pseudo-random numbers
    /// that RNGState::generator can produce. Any fixed sequence has
    /// fairly large period (\math{2^{132},} with RandBLAS' default) and
    /// is statistically independent from sequences induced by different keys.
    ///
    /// To increment the key by "step," call \math{\ttt{key.incr(step)}}.
    typename RNG::key_type key;

    const static int len_c = RNG::ctr_type::static_size;
    static_assert(len_c >= 2);
    const static int len_k = RNG::key_type::static_size;

    /// Initialize the counter and key to zero.
    RNGState() : counter{}, key{} {}

    /// Initialize the counter and key to zero, then increment the key by k.
    RNGState(uint64_t k) : counter{}, key{} { key.incr(k); }

    // construct from a key
    RNGState(key_type const &k) : counter{}, key(k) {}

    // Initialize counter and key arrays at the given values.
    RNGState(ctr_type const &c, key_type const &k) : counter(c), key(k) {}

    // move construct from an initial counter and key
    RNGState(ctr_type &&c, key_type &&k) : counter(std::move(c)), key(std::move(k)) {}

    // move constructor.
    RNGState(RNGState<RNG> &&s) : RNGState(std::move(s.counter), std::move(s.key)) {};

    ~RNGState() {};

    /// Copy constructor.
    RNGState(const RNGState<RNG> &s) : RNGState(s.counter, s.key) {};

    // A copy-assignment operator.
    RNGState<RNG> &operator=(const RNGState<RNG> &s) {
        std::memcpy(this->counter.v, s.counter.v, this->len_c * sizeof(ctr_uint));
        std::memcpy(this->key.v,     s.key.v,     this->len_k * sizeof(key_uint));
        return *this;
    };

    //
    // Comparators (for now, these are just for testing and debugging)
    // 

    bool operator==(const RNGState<RNG> &s) const {
        // the compiler should only allow comparisons between RNGStates of the same type.
        for (int i = 0; i < len_c; ++i) {
            if (counter.v[i] != s.counter.v[i]) { return false; }
        }
        for (int i = 0; i < len_k; ++i) {
            if (key.v[i] != s.key.v[i]) { return false; }
        }
        return true;
    };

    bool operator!=(const RNGState<RNG> &s) const {
        return !(*this == s);
    };

};

template <typename RNG>
const int RandBLAS::RNGState<RNG>::len_c;

template <typename RNG>
const int RandBLAS::RNGState<RNG>::len_k;

template <typename RNG>
std::ostream &operator<<(
    std::ostream &out,
    const RNGState<RNG> &s
) {
    int i;
    out << "counter : {";
    for (i = 0; i < s.len_c - 1; ++i) {
        out << s.counter[i] << ", ";
    }
    out << s.counter[i] << "}\n";
    out << "key     : {";
    for (i = 0; i < s.len_k - 1; ++i) {
        out << s.key[i] << ", ";
    }
    out << s.key[i] << "}";
    return out;
}

/**
 * Stores stride information for a matrix represented as a buffer.
 * The intended semantics for a buffer "A" and the conceptualized
 * matrix "mat(A)" are 
 * 
 *  mat(A)_{ij} == A[i * inter_row_stride + j * inter_col_stride].
 * 
 * for all (i, j) within the bounds of mat(A).
 */
struct stride_64t {
    int64_t inter_row_stride; // step down a column
    int64_t inter_col_stride; // step along a row
};

inline stride_64t layout_to_strides(blas::Layout layout, int64_t ldim) {
    if (layout == blas::Layout::ColMajor) {
        return stride_64t{(int64_t) 1, ldim};
    } else {
        return stride_64t{ldim, (int64_t) 1};
    }
}

inline stride_64t layout_to_strides(blas::Layout layout, int64_t n_rows, int64_t n_cols) {
    if (layout == blas::Layout::ColMajor) {
        return stride_64t{(int64_t) 1, n_rows};
    } else {
        return stride_64t{n_cols, (int64_t) 1};
    }
}

struct dims64_t {
    int64_t n_rows;
    int64_t n_cols;
};

inline dims64_t dims_before_op(int64_t m, int64_t n, blas::Op op) {
    if (op == blas::Op::NoTrans) {
        return {m, n};
    } else {
        return {n, m};
    }
}

struct submat_spec_64t {
    int64_t pointer_offset;
    int64_t ldim;
};

inline submat_spec_64t offset_and_ldim(
    blas::Layout layout, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s
) {
    if (layout == blas::Layout::ColMajor) {
        int64_t offset = ro_s + n_rows * co_s;
        return submat_spec_64t{offset, n_rows};
    } else {
        int64_t offset = ro_s * n_cols + co_s;
        return submat_spec_64t{offset, n_cols};
    }
}


#ifdef __cpp_concepts
template<typename T>
concept SignedInteger = (std::numeric_limits<T>::is_signed && std::numeric_limits<T>::is_integer);
#else
#define SignedInteger typename
#endif


template <SignedInteger TI, SignedInteger TO = int64_t>
inline TO safe_int_product(TI a, TI b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    TO c = a * b;
    TO b_check = c / a;
    TO a_check = c / b;
    if ((a_check != a) || (b_check != b)) {
        std::stringstream s;
        s << "Overflow when multiplying a (=" << a << ") and b(=" << b << "), which resulted in " << c << ".\n";
        throw std::overflow_error(s.str());
    }
    return c;
}


// ---------------------------------------------------------------------------
/// Sketching operators are only "useful" for dimension reduction if they're
/// non-square.
///
/// The larger dimension of a sketching operator has a different
/// semantic role than the small dimension. This enum provides a way for us
/// to refer to the larger or smaller dimension in a way that's agnostic to 
/// whether the sketching operator is wide or tall.
///  
/// For a wide matrix, its *short-axis vectors* are its columns, and its
/// *long-axis vectors* are its rows.
///
/// For a tall matrix, its short-axis vectors are its rows, and its
/// long-axis vectors are its columns.
///
enum class Axis : char {
    // ---------------------------------------------------------------------------
    Short = 'S',

    // ---------------------------------------------------------------------------
    Long = 'L'
};

// ---------------------------------------------------------------------------
/// Returns max(n_rows, n_cols) if major_axis == Axis::Long, and returns 
/// min(n_rows, n_cols) otherwise.
///
inline int64_t get_dim_major(Axis major_axis, int64_t n_rows, int64_t n_cols) {
    if (major_axis == Axis::Long) {
        return std::max(n_rows, n_cols);
    } else {
        return std::min(n_rows, n_cols);
    }
}


#ifdef __cpp_concepts
// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
/// **Mathematical description**
///
/// Matrices sampled from sketching distributions in RandBLAS are mean-zero
/// and have covariance matrices that are proportional to the identity.
///
/// Formally, 
/// if :math:`\D` is a distribution over :math:`r \times c` matrices and 
/// :math:`\mtxS` is a sample from :math:`\D`,  then
/// :math:`\mathbb{E}\mtxS = \mathbf{0}_{r \times c}` and
///
/// .. math::
///    :nowrap:
///     
///     \begin{gather}
///     \theta^2 \cdot \mathbb{E}\left[ \mtxS^T\mtxS \right]=\mathbf{I}_{c \times c}& \nonumber \\
///     \,\phi^2 \cdot \mathbb{E}\left[ \mtxS{\mtxS}^T\, \right]=\mathbf{I}_{r \times r}& \nonumber
///     \end{gather}
///
/// hold for some :math:`\theta > 0` and :math:`\phi > 0`.
///
/// The *isometry scale* of the distribution
/// is :math:`\alpha := \theta` if :math:`c \geq r` and :math:`\alpha := \phi` otherwise. If you want to
/// sketch in a way that preserves squared norms in expectation, then you should sketch with 
/// a scaled sample :math:`\alpha \mtxS` rather than the sample itself.
///
/// **Programmatic description**
///
/// A variable :math:`\ttt{D}` of a type that conforms to the 
/// :math:`\ttt{SketchingDistribution}` concept has the following attributes.
///
/// .. list-table::
///    :widths: 25 30 40
///    :header-rows: 1
///    
///    * - 
///      - type
///      - description
///    * - :math:`\ttt{D.n_rows}`
///      - :math:`\ttt{const int64_t}`
///      - samples from :math:`\ttt{D}` have this many rows
///    * - :math:`\ttt{D.n_cols}`
///      - :math:`\ttt{const int64_t}`
///      - samples from :math:`\ttt{D}` have this many columns
///    * - :math:`\ttt{D.isometry_scale}`
///      - :math:`\ttt{const double}`
///      - See above.
///
/// Note that the isometry scale is always stored in double precision; this has no bearing 
/// on the precision of sketching operators that are sampled from a :math:`\ttt{SketchingDistribution}`.
///
/// **Notes**
///
/// RandBLAS has two SketchingDistribution types: DenseDist and SparseDist.
/// These types have members called called "major_axis,"
/// "dim_major," and "dim_minor." These members have similar semantic roles across
/// the two classes, but their precise meanings differ significantly.
/// @endverbatim
template<typename SkDist>
concept SketchingDistribution = requires(SkDist D) {
    { D.n_rows }     -> std::same_as<const int64_t&>;
    { D.n_cols }     -> std::same_as<const int64_t&>;
    { D.isometry_scale } -> std::same_as<const double&>;
};
#else
#define SketchingDistribution typename
#endif


#ifdef __cpp_concepts
// =============================================================================
/// A type \math{\ttt{SKOP}} that conforms to the SketchingOperator concept
/// has three member types.
/// @verbatim embed:rst:leading-slashes
///
/// .. list-table::
///    :widths: 25 65
///    :header-rows: 0
///
///    * - :math:`\ttt{SKOP::distribution_t}`
///      - A type conforming to the SketchingDistribution concept.
///    * - :math:`\ttt{SKOP::state_t}`
///      - A template instantiation of RNGState.
///    * - :math:`\ttt{SKOP::scalar_t}`
///      - Real scalar type used in matrix representations of :math:`\ttt{SKOP}\text{s}.`
///
/// And an object :math:`\ttt{S}` of type :math:`\ttt{SKOP}` has the following 
/// instance members.
///
/// .. list-table::
///    :widths: 20 25 45
///    :header-rows: 0
///    
///    * - :math:`\ttt{S.dist}`
///      - :math:`\ttt{const distribution_t}`
///      - Distribution from which this operator is sampled.
///    * - :math:`\ttt{S.n_rows}`
///      - :math:`\ttt{const int64_t}`
///      - An alias for :math:`\ttt{S.dist.n_rows}.`
///    * - :math:`\ttt{S.n_cols}`
///      - :math:`\ttt{const int64_t}`
///      - An alias for :math:`\ttt{S.dist.n_cols}.`
///    * - :math:`\ttt{S.seed_state}`
///      - :math:`\ttt{const state_t}`
///      - RNGState used to construct
///        an explicit representation of :math:`\ttt{S}`.
///    * - :math:`\ttt{S.next_state}`
///      - :math:`\ttt{const state_t}`
///      - An RNGState that can be used in a call to a random sampling routine
///        whose output should be statistically independent from :math:`\ttt{S}.`   
///    * - :math:`\ttt{S.own_memory}`
///      - :math:`\ttt{bool}`
///      - A flag used to indicate whether internal functions
///        have permission to attach memory to :math:`\ttt{S},`
///        *and* whether the destructor of :math:`\ttt{S}` has the
///        responsibility to delete any memory that's attached to
///        :math:`\ttt{S}.`
///
/// 
/// RandBLAS only has two SketchingOperator types: DenseSkOp and SparseSkOp. These types
/// have several things in common
/// that aren't enforced by the SketchingOperator concept. Most notably, they have 
/// constructors of the following form.
///
/// .. code:: c++
///
///    SKOP(distribution_t dist, state_t seed_state) 
///     : dist(dist), 
///       seed_state(seed_state), 
///       next_state(/* type-specific function of state and dist */), 
///       n_rows(dist.n_rows), 
///       n_cols(dist.n_cols), 
///       own_memory(true)
///       /* type-specific initializers */ { };
///
/// @endverbatim
template<typename SKOP>
concept SketchingOperator = requires {
    typename SKOP::distribution_t;
    typename SKOP::state_t;
    typename SKOP::scalar_t;
} && SketchingDistribution<typename SKOP::distribution_t> && requires(
    SKOP S,typename SKOP::distribution_t dist, typename SKOP::state_t state
) {
    { S.dist }       -> std::same_as<const typename SKOP::distribution_t&>;
    { S.n_rows }     -> std::same_as<const int64_t&>;
    { S.n_cols }     -> std::same_as<const int64_t&>;
    { S.seed_state } -> std::same_as<const typename SKOP::state_t&>;
    { S.next_state } -> std::same_as<const typename SKOP::state_t&>;
    { S.own_memory } -> std::same_as<bool&>;
};
#else
#define SketchingOperator typename
#endif

} // end namespace RandBLAS::base

