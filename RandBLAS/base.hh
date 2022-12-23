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

    /// @name conversions
    ///{
    RNGState(std::tuple<typename RNG::ctr_type, typename RNG::key_type> const& tup) : counter(std::get<0>(tup)), key(std::get<1>(tup)) {}
    RNGState(std::tuple<typename RNG::ctr_type, typename RNG::key_type> && tup) : counter(std::move(std::get<0>(tup))), key(std::move(std::get<1>(tup))) {}
    operator std::tuple<typename RNG::ctr_type const&, typename RNG::key_type const&> () const { return std::tie(std::as_const(counter), std::as_const(key)); }
    operator std::tuple<typename RNG::ctr_type&, typename RNG::key_type&> () { return std::tie(counter, key); }
    ///}

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


/** Apply boxmuller transform to all elements of r. The number of elements of r
 * must be evenly divisible by 2.
 *
 * @tparam RNG a random123 CBRNG type
 * @tparam T the desired return type. The default return type is dictated by
 *           the RNG's counter element type. float for 32 bit counter elements
 *           and double for 64.
 *
 * @param[in] ri a sequence of n random values generated using random123 CBRNG
 *               type RNG. The transform is applied pair wise.
 *
 * @returns n transformed floating point values.
 */
template <typename RNG, typename T = typename std::conditional
    <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
auto boxmulall(
    typename RNG::ctr_type const& ri
) {
    std::array<T, RNG::ctr_type::static_size> ro;
    int nit = RNG::ctr_type::static_size / 2;
    for (int i = 0; i < nit; ++i)
    {
        auto [v0, v1] = r123::boxmuller(ri[2*i], ri[2*i + 1]);
        ro[2*i    ] = v0;
        ro[2*i + 1] = v1;
    }
    return ro;
}

/// Generate a sequence of random values and apply a Box-Muller transform.
struct boxmul
{
    /** Generate a sequence of random values and apply a Box-Muller transform.
     *
     * @tparam RNG a random123 CBRNG type
     * @tparam T the desired return type. The default return type is dictated by
     *           the RNG's counter element type. float for 32 bit counter elements
     *           and double for 64.
     *
     * @param[in] a random123 CBRNG instance used to generate the sequence
     * @param[in] the CBRNG counter of n elements
     * @param[in] the CBRNG key
     *
     * @returns the generated and transformed sequence of n floating point
     *          values. The return type is dictated by the RNG's counter
     *          element type. float for 32 bit counter elements and double for
     *          64.
     */
    template <typename RNG>
    static
    auto generate(
        RNG &rng,
        typename RNG::ctr_type const& c,
        typename RNG::key_type const& k
    ) {
        return boxmulall<RNG>(rng(c,k));
    }
};

/// Generate a sequence of random values and transform to -1.0 to 1.0.
struct uneg11
{
    /** Generate a sequence of random values and transform to -1.0 to 1.0.
     *
     * @tparam RNG a random123 CBRNG type
     * @tparam T the desired return type. The default return type is dictated by
     *           the RNG's counter element type. float for 32 bit counter elements
     *           and double for 64.
     *
     * @param[in] a random123 CBRNG instance used to generate the sequence
     * @param[in] the CBRNG counter of n elements
     * @param[in] the CBRNG key
     *
     * @returns the generated and transformed sequence of n floating point
     *          values. The return type is dictated by the counter element
     *          type. float for 32 bit counter elements and double for 64.
     */
    template <typename RNG, typename T = typename std::conditional
        <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
    static
    auto generate(
        RNG &rng,
        typename RNG::ctr_type const& c,
        typename RNG::key_type const& k
    ) {
        return r123::uneg11all<T>(rng(c,k));
    }
};
}; // end namespace RandBLAS::base

#endif
