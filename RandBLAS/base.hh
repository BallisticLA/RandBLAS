#ifndef randblas_base_hh
#define randblas_base_hh

/// @file

#include "RandBLAS/config.h"
#include "RandBLAS/random_gen.hh"

#include <tuple>
#include <utility>
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <iostream>

#define RandBLAS_HAS_OpenMP
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#include<iostream>

/// code common across the project
namespace RandBLAS::base {

/// Enumerate the names of the Random123 CBRNGs
enum class RNGName : char {None = '\0', Philox = 'P', Threefry = 'T'};


/** A CBRNG state consiting of a counter and a key.
 * @tparam RNG One of Random123 CBRNG's e.g. Philox4x32
 */
template <typename RNG = r123::Philox4x32>
struct RNGState
{
    using generator = RNG;

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

    /// construct integer values
    RNGState(typename RNG::ctr_type::value_type c, typename RNG::ukey_type::value_type k) : counter{{c}}, key{{k}} {}

    typename RNG::ctr_type counter; ///< the counter
    typename RNG::key_type key;     ///< the key
};


/// serialize the state to a stream
template <typename RNG>
std::ostream &operator<<(
    std::ostream &out,
    const RNGState<RNG> &s
) {
    out << "counter : {" << s.counter << "}" << std::endl
        << "key     : {" << s.key << "}" << std::endl;
    return out;
}

} // end namespace RandBLAS::base

#endif
