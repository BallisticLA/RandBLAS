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
#include <tuple>
#include <utility>
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <iostream>

#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#include<iostream>
#include<numeric>


/// code common across the project
namespace RandBLAS {


/** A representation of the state of a counter-based random number generator
 * (CBRNG) defined in Random123. The representation consists of two arrays:
 * the counter and the key. The arrays' types are statically sized, small
 * (typically of length 2 or 4), and can be distinct from one another.
 * 
 * The template parameter RNG is a CBRNG type in defined in Random123. We've found
 * that Philox-based CBRNGs work best for our purposes, but we also support Threefry-based CBRNGS.
 */
template <typename RNG = r123::Philox4x32>
struct RNGState {
    using generator = RNG;
    
    using ctr_type = typename RNG::ctr_type;
    // ^ An array type defined in Random123.
    using key_type = typename RNG::key_type;
    // ^ An array type defined in Random123.
    using ctr_uint = typename RNG::ctr_type::value_type;
    // ^ The unsigned integer type used in this RNGState's counter array.

    /// -------------------------------------------------------------------
    /// @brief The unsigned integer type used in this RNGState's key array.
    ///        This is typically std::uint32_t, but it can be std::uint64_t.
    using key_uint = typename RNG::key_type::value_type;


    const static int len_c = RNG::ctr_type::static_size;
    const static int len_k = RNG::key_type::static_size;
    typename RNG::ctr_type counter;
    // ^ This RNGState's counter array.

    /// ------------------------------------------------------------------
    /// This RNGState's key array. If you want to manually advance the key
    /// by an integer increment of size "step," then you do so by calling 
    /// this->key.incr(step).
    typename RNG::key_type key;


    /// Initialize the counter and key arrays to all zeros.
    RNGState() : counter{{0}}, key(key_type{{}}) {}

    // construct from a key
    RNGState(key_type const &k) : counter{{0}}, key(k) {}

    // Initialize counter and key arrays at the given values.
    RNGState(ctr_type const &c, key_type const &k) : counter(c), key(k) {}

    // move construct from an initial counter and key
    RNGState(ctr_type &&c, key_type &&k) : counter(std::move(c)), key(std::move(k)) {}

    /// Initialize the counter array to all zeros. Initialize the key array to have first
    /// element equal to k and all other elements equal to zero.
    RNGState(key_uint k) : counter{{0}}, key{{k}} {}

    ~RNGState() {};

    /// A copy constructor.
    RNGState(const RNGState<RNG> &s);

    RNGState<RNG> &operator=(const RNGState<RNG> &s);

};


template <typename RNG>
RNGState<RNG>::RNGState(
    const RNGState<RNG> &s
) {
    std::memcpy(this->counter.v, s.counter.v, this->len_c * sizeof(ctr_uint));
    std::memcpy(this->key.v,     s.key.v,     this->len_k * sizeof(key_uint));
}

template <typename RNG>
RNGState<RNG> &RNGState<RNG>::operator=(
    const RNGState &s
) {
    std::memcpy(this->counter.v, s.counter.v, this->len_c * sizeof(ctr_uint));
    std::memcpy(this->key.v,     s.key.v,     this->len_k * sizeof(key_uint));
    return *this;
}

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
/// @verbatim embed:rst:leading-slashes
///
/// *Note: This concept does not apply to distributions over square matrices.*
///
/// We define sketching distributions in a way that is agnostic to sketching from the left or from the right.
/// This requires thinking less about the properties of a sketching operator's rows and columns
/// and more about the properties of the vectors along their *longer axis* and *shorter axis*,
/// in the sense of the table below.
///
/// .. list-table::
///    :widths: 34 33 33
///    :header-rows: 1
///
///    * - When sketching from the ...
///      - the short-axis vectors are ...
///      - the long-axis vectors are ...
///    * - left, with a wide operator
///      - the operator's columns
///      - the operator's rows
///    * - right, with a tall operator
///      - the operator's rows
///      - the operator's columns
///
/// If a distribution has major axis :math:`M` then its samples will have statistically
/// independent :math:`M`-axis vectors. A major axis is *undefined* when there are dependencies across both 
/// rows and columns.
///
/// @endverbatim
enum class MajorAxis : char {
    // ---------------------------------------------------------------------------
    Short = 'S',

    // ---------------------------------------------------------------------------
    Long = 'L',

    // ---------------------------------------------------------------------------
    Undefined = 'U'
};


#ifdef __cpp_concepts
// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
/// **Mathematical description**
///
/// Matrices sampled from sketching distributions in RandBLAS are mean-zero
/// and have covariance matrices that are proportional to the identity. That is, 
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
///    * - :math:`\ttt{D.major_axis}`
///      - :math:`\ttt{const MajorAxis}`
///      - Implementation-dependent; see MajorAxis documentation.
///    * - :math:`\ttt{D.isometry_scale}`
///      - :math:`\ttt{const double}`
///      - See above.
///
/// Note that the isometry scale is always stored in double precision; this has no bearing 
/// on the precision of sketching operators that are sampled from a :math:`\ttt{SketchingDistribution}`.
///
/// @endverbatim
template<typename SkDist>
concept SketchingDistribution = requires(SkDist D) {
    { D.n_rows }     -> std::same_as<const int64_t&>;
    { D.n_cols }     -> std::same_as<const int64_t&>;
    { D.major_axis } -> std::same_as<const MajorAxis&>;
    { D.isometry_scale } -> std::same_as<const double&>;
};
#else
#define SketchingDistribution typename
#endif


#ifdef __cpp_concepts
// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
/// .. NOTE: \ttt expands to \texttt (its definition is given in an rst file)
///
/// Words. Hello!
///
/// @endverbatim
template<typename SKOP>
concept SketchingOperator = requires(SKOP S) {
    { S.n_rows }     -> std::same_as<const int64_t&>;
    { S.n_cols }     -> std::same_as<const int64_t&>;
    { S.dist   }     -> SketchingDistribution;
    { S.seed_state } -> std::same_as<const typename SKOP::state_t&>;
    { S.next_state } -> std::same_as<const typename SKOP::state_t&>;
};
#else
#define SketchingOperator typename
#endif


} // end namespace RandBLAS::base
