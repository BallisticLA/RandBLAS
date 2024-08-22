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

/**
 * Stores stride information for a matrix represented as a buffer.
 * The intended semantics for a buffer "A" and the conceptualized
 * matrix "mat(A)" are 
 * 
 *  mat(A)[i, j] == A[i * inter_row_stride + j * inter_col_stride].
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


template<typename T>
concept SignedInteger = (std::numeric_limits<T>::is_signed && std::numeric_limits<T>::is_integer);


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


enum class MajorAxis : char {
    // ---------------------------------------------------------------------------
    ///  short-axis vectors (cols of a wide matrix, rows of a tall matrix)
    Short = 'S',

    // ---------------------------------------------------------------------------
    ///  long-axis vectors (rows of a wide matrix, cols of a tall matrix)
    Long = 'L',

    // ---------------------------------------------------------------------------
    ///  Undefined (used when row-major vs column-major must be explicit)
    Undefined = 'U'
};


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

// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
/// .. NOTE: \ttt expands to \texttt (its definition is given in an rst file)
///
/// Words. Hello!
///
/// @endverbatim
template<typename SkDist>
concept SketchingDistribution = requires(SkDist D) {
    { D.n_rows }     -> std::convertible_to<const int64_t>;
    { D.n_cols }     -> std::convertible_to<const int64_t>;
    { D.major_axis } -> std::convertible_to<const MajorAxis>;
};

// =============================================================================
/// \fn isometry_scale_factor(SkDist D) 
/// @verbatim embed:rst:leading-slashes
/// Words here ...
/// @endverbatim
template <typename T, SketchingDistribution SkDist>
inline T isometry_scale_factor(SkDist D);


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
    { S.n_rows }     -> std::convertible_to<const int64_t>;
    { S.n_cols }     -> std::convertible_to<const int64_t>;
    { S.seed_state } -> std::convertible_to<const typename SKOP::state_t>;
    { S.seed_state } -> std::convertible_to<const typename SKOP::state_t>;
};


} // end namespace RandBLAS::base
