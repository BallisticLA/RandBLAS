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
#include <cstdint>
#include <iostream>

#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#include<iostream>


/// code common across the project
namespace RandBLAS {

using std::uint64_t;
template <rng::CounterBasedEngine Engine>
    requires requires(RNGState<Engine> const& state, std::ostream& stream) {
        state.counter.size();
        state.key.size();
        state.counter[0];
        state.key[0];
        stream << state.counter[0];
        stream << state.key[0];
    }
std::ostream &operator<<(
    std::ostream &out,
    const RNGState<Engine> &s
) {
    auto const& counter = s.counter;
    auto const& key = s.key;
    out << "counter : {";
    for (std::size_t i = 0; i + 1 < counter.size(); ++i) {
        out << counter[i] << ", ";
    }
    out << counter[counter.size() - 1] << "}\n";
    out << "key     : {";
    for (std::size_t i = 0; i + 1 < key.size(); ++i) {
        out << key[i] << ", ";
    }
    out << key[key.size() - 1] << "}";
    return out;
}

inline blas::Layout flipped_layout(const blas::Layout &layout_before) {
    using blas::Layout;
    return (layout_before == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor;
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


template <typename ordinal_t>
int64_t cast_int64t(ordinal_t arg_n) {
    auto n = static_cast<int64_t>(arg_n);
    bool lossy_cast = (double)n != (double)arg_n;
    if (lossy_cast) {
        std::cerr << std::endl;
        std::cerr << "A floating point number `arg_n` has been passed as a dimensional parameter,"<< std::endl;
        std::cerr << "where floor(arg_n) < arg_n. We round dimensions down in these situations." << std::endl;
        std::cerr << "Avoid this warning by providing integer arguments." << std::endl << std::endl;
    }
    return n;
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
/// 
/// These types also have instance methods DenseDist::sample and SparseDist::sample,
/// which take a scalar template parameter and an RNGState object and return 
/// DenseSkOps and SparseSkOps, respectively. This makes it easy to sample a
/// sketching operator from a distribution :math:`\ttt{D}` without knowing the
/// distribution's type:
///
/// .. code:: c++
///
///    auto S = D.sample<double>(seed);
/// 
/// 
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
