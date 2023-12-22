#ifndef randblas_base_hh
#define randblas_base_hh

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

struct stride_64t {
    int64_t inter_row_stride;
    int64_t inter_col_stride;
};

static inline stride_64t layout_to_strides(blas::Layout layout, int64_t ldim) {
    if (layout == blas::Layout::ColMajor) {
        return stride_64t{(int64_t) 1, ldim};
    } else {
        return stride_64t{ldim, (int64_t) 1};
    }
}

struct dims64_t {
    int64_t n_rows;
    int64_t n_cols;
};

static inline dims64_t dims_before_op(int64_t m, int64_t n, blas::Op op) {
    if (op == blas::Op::NoTrans) {
        return {m, n};
    } else {
        return {n, m};
    }
}


template<typename T>
concept SignedInteger = (std::numeric_limits<T>::is_signed && std::numeric_limits<T>::is_integer);


enum class MajorAxis : char {
    // ---------------------------------------------------------------------------
    ///  short-axis vectors (cols of a wide matrix, rows of a tall matrix)
    Short = 'S',

    // ---------------------------------------------------------------------------
    ///  long-axis vectors (rows of a wide matrix, cols of a tall matrix)
    Long = 'L'
};

/// Enumerate the names of the Random123 CBRNGs
enum class RNGName : char {None = '\0', Philox = 'P', Threefry = 'T'};


/** A CBRNG state consiting of a counter and a key.
 * @tparam RNG One of Random123 CBRNG's e.g. Philox4x32
 */
template <typename RNG = r123::Philox4x32>
struct RNGState
{
    using generator = RNG;
    using ctr_value_type = typename RNG::ctr_type::value_type;
    using key_value_type = typename RNG::ukey_type::value_type;
    using ctr_type = typename RNG::ctr_type;
    using key_type = typename RNG::key_type;

    const static int len_c = RNG::ctr_type::static_size;
    const static int len_k = RNG::key_type::static_size;
    ctr_type counter; ///< the counter
    key_type key;     ///< the key

    /// default construct both counter and key are zero'd
    RNGState() : counter{{}}, key(typename RNG::ukey_type{{}}) {}

    /** construct with a seed. the seed is stored in the key.
     * @param[in] k a key value to use as a seed
     */
    RNGState(typename RNG::ukey_type const& k) : counter{{}}, key(k) {}

    /// construct from an initial counter and key
    RNGState(typename RNG::ctr_type const& c, typename RNG::key_type const& k) : counter(c), key(k) {}

    /// move construct from an initial counter and key
    RNGState(typename RNG::ctr_type &&c, typename RNG::key_type &&k) : counter(std::move(c)), key(std::move(k)) {}

    /// construct from an integer key
    RNGState(typename RNG::ukey_type::value_type k) : counter{{0}}, key{{k}} {}

    ~RNGState() {};

    RNGState(const RNGState<RNG> &s);

    RNGState<RNG> &operator=(const RNGState<RNG> &s);

};


template <typename RNG>
RNGState<RNG>::RNGState(
    const RNGState<RNG> &s
) {
    std::memcpy(this->counter.v, s.counter.v, this->len_c * sizeof(ctr_value_type));
    std::memcpy(this->key.v,     s.key.v,     this->len_k * sizeof(key_value_type));
}

template <typename RNG>
RNGState<RNG> &RNGState<RNG>::operator=(
    const RNGState &s
) {
    std::memcpy(this->counter.v, s.counter.v, this->len_c * sizeof(ctr_value_type));
    std::memcpy(this->key.v,     s.key.v,     this->len_k * sizeof(key_value_type));
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

} // end namespace RandBLAS::base

#endif
