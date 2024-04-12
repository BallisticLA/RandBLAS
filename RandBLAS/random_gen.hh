#ifndef random_gen_hh
#define random_gen_hh

/// @file

#include <Random123/features/compilerfeatures.h>

// this is for sincosf
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include <math.h>

#if !defined(R123_NO_SINCOS) && defined(__APPLE__)
/* MacOS X 10.10.5 (2015) doesn't have sincosf */
// use "-D __APPLE__" as a compiler flag to make sure this is hit.
#define R123_NO_SINCOS 1
#endif

#if R123_NO_SINCOS /* enable this if sincos and sincosf are not in the math library */
R123_CUDA_DEVICE R123_STATIC_INLINE void sincosf(float x, float *s, float *c) {
    *s = sinf(x);
    *c = cosf(x);
}

R123_CUDA_DEVICE R123_STATIC_INLINE void sincos(double x, double *s, double *c) {
    *s = sin(x);
    *c = cos(x);
}
#endif /* sincos is not in the math library */

// this is for sincosf
#if !defined(__CUDACC__)
static inline void sincospif(float x, float *s, float *c) {
    const float PIf = 3.1415926535897932f;
    sincosf(PIf*x, s, c);
}

static inline void sincospi(double x, double *s, double *c) {
    const double PI = 3.1415926535897932;
    sincos(PI*x, s, c);
}
#endif

#include <Random123/array.h>
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/ars.h>
// NOTE: we do not support Random123's AES generator.
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>

/// our extensions to random123
namespace r123ext
{
/** Apply boxmuller transform to all elements of ri. The number of elements of r
 * must be evenly divisible by 2. See also r123::uneg11all.
 *
 * @tparam CTR a random123 CBRNG ctr_type
 * @tparam T the return element type. The default return type is dictated by
 *           the RNG's ctr_type's value_type : float for 32 bit counter elements
 *           and double for 64.
 *
 * @param[in] ri a sequence of N random values generated using random123 CBRNG
 *               type RNG. The transform is applied pair wise to the sequence.
 *
 * @returns a std::array<T,N> of transformed floating point values.
 */
template <typename CTR, typename T = typename std::conditional
    <sizeof(typename CTR::value_type) == sizeof(uint32_t), float, double>::type>
auto boxmulall(
    CTR const &ri
) {
    std::array<T, CTR::static_size> ro;
    int nit = CTR::static_size / 2;
    for (int i = 0; i < nit; ++i)
    {
        auto [v0, v1] = r123::boxmuller(ri[2*i], ri[2*i + 1]);
        ro[2*i    ] = v0;
        ro[2*i + 1] = v1;
    }
    return ro;
}

/** @defgroup generators
 * Generators take CBRNG, counter,and key instances and return a sequence of
 * random floating point numbers in a std::array. The length of the squence is
 * the length of the counter and the precision is float for 32 bit counters and
 * double for 64.
 */
/// @{

/// Generate a sequence of random values and apply a Box-Muller transform.
struct boxmul
{
    /** Generate a sequence of random values and apply a Box-Muller transform.
     *
     * @tparam RNG a random123 CBRNG type
     *
     * @param[in] a random123 CBRNG instance used to generate the sequence
     * @param[in] the CBRNG counter
     * @param[in] the CBRNG key
     *
     * @returns a std::array<N,T> where N is the CBRNG's ctr_type::static_size
     *          and T is deduced from the RNG's counter element type : float
     *          for 32 bit counter elements and double for 64. For example when
     *          RNG is Philox4x32 the return is a std::array<float,4>.
     */
    template <typename RNG>
    static
    auto generate(
        RNG &rng,
        typename RNG::ctr_type const &c,
        typename RNG::key_type const &k
    ) {
        return boxmulall(rng(c,k));
    }
};

/// Generate a sequence of random values and transform to -1.0 to 1.0.
struct uneg11
{
    /** Generate a sequence of random values and transform to -1.0 to 1.0.
     *
     * @tparam RNG a random123 CBRNG type
     *
     * @param[in] a random123 CBRNG instance used to generate the sequence
     * @param[in] the CBRNG counter
     * @param[in] the CBRNG key
     *
     * @returns a std::array<N,T> where N is the CBRNG's ctr_type::static_size
     *          and T is deduced from the RNG's counter element type : float
     *          for 32 bit counter elements and double for 64. For example when
     *          RNG is Philox4x32 the return is a std::array<float,4>.
     */
    template <typename RNG, typename T = typename std::conditional
        <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
    static
    auto generate(
        RNG &rng,
        typename RNG::ctr_type const &c,
        typename RNG::key_type const &k
    ) {
        return r123::uneg11all<T>(rng(c,k));
    }
};

/// @}

} // end of namespace r123ext

#endif
