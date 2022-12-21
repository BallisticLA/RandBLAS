
#include <Random123/boxmuller.hpp>
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/aes.h>
#include <Random123/ars.h>

#include <tuple>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <utility>
#include <chrono>
#include <type_traits>

#define USE_OMP
#if defined(USE_OMP)
#include <omp.h>
#endif




/// extend r123::float2 to work with structured bindings
namespace std {
  template<> struct tuple_size<r123::float2> { static constexpr size_t value = 2; };
  template<> struct tuple_element<0, r123::float2> { using type = float; };
  template<> struct tuple_element<1, r123::float2> { using type = float; };
}

namespace r123 {
template<std::size_t I>
std::tuple_element_t<I, r123::float2> get(r123::float2 const& f2)
{
  if constexpr (I == 0) return f2.x;
  if constexpr (I == 1) return f2.y;
}

}




/** CBRNG state.
 * @param RNG One of Random123 CBRNG's e.g. Philox4x32
 */
template <typename RNG>
struct RNGState
{
    using generator = RNG;

    /// default construct both counter and key are zero'd
    RNGState() : counter{{}}, key(typename RNG::ukey_type{{}}) {}

    /** construct with a seed
     * @param[in] k a key value to use as a seed
     */
    RNGState(typename RNG::ukey_type const& k) : counter{{}}, key(k) {}

    /// construct from an initial counter and key
    RNGState(typename RNG::ctr_type const& c, typename RNG::key_type const& k) : counter(c), key(k) {}

    /// construct from an initial counter and key
    RNGState(typename RNG::ctr_type &&c, typename RNG::key_type &&k) : counter(std::move(c)), key(std::move(k)) {}

    /// @name conversions
    ///{
    RNGState(std::tuple<typename RNG::ctr_type, typename RNG::key_type> const& tup) : counter(std::get<0>(tup)), key(std::get<1>(tup)) {}
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


/** apply boxmuller transform to all elements of r. The number of elements of r
 * must be evenly divisible by 2.
 */
template <typename RNG, typename T = typename std::conditional
        <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
auto boxmulall(typename RNG::ctr_type const& ri)
{
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

/// generate a sequence of random values and apply a Box-Muller transform
struct boxmul
{
    template <typename RNG>
    static
    auto generate(RNG &rng, typename RNG::ctr_type const& c, typename RNG::key_type const& k)
    {
        return boxmulall<RNG>(rng(c,k));
    }
};

/// generate a sequence of random values and transform to -1.0 to 1.0
struct uneg11
{
    template <typename RNG, typename T = typename std::conditional
            <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
    static
    auto generate(RNG &rng, typename RNG::ctr_type const& c, typename RNG::key_type const& k)
    {
        return r123::uneg11all<T>(rng(c,k));
    }
};


/**
 */
template <typename T, typename RNG, typename OP>
auto fill_rmat(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState<RNG> & seed
) {
    RNG rng;
    auto [c, k] = seed;

    int64_t dim = n_rows * n_cols;
    int64_t nit = dim / RNG::ctr_type::static_size;
    int64_t nlast = dim % RNG::ctr_type::static_size;

#if defined(USE_OMP)
    #pragma omp parallel firstprivate(c, k)
    {
        // add the start index to the counter in order to make the sequence
        // deterministic independent of the number of threads.
        int ti = omp_get_thread_num();
        int nt = omp_get_num_threads();

        int64_t chs = nit / nt;
        int64_t nlg = nit % nt;
        int64_t i0 = chs * ti + (ti < nlg ? ti : nlg);
        int64_t i1 = i0 + chs + (ti < nlg ? 1 : 0);

        auto cc = c; // because of pointers used internal to RNG::ctr_type

        cc.incr(i0);
#else
        int64_t i0 = 0;
        int64_t i1 = nit;
#endif
        for (int64_t i = i0; i < i1; ++i)
        {
            auto rv = OP::generate(rng, cc, k);

            for (int j = 0; j < RNG::ctr_type::static_size; ++j)
            {
               mat[RNG::ctr_type::static_size * i + j] = rv[j];
            }

            cc.incr();
        }
#if defined(USE_OMP)
    }
    // puts the counter in the correct state when threads are used.
    c.incr(nit);
#endif

    if (nlast)
    {
        auto rv = OP::generate(rng, c, k);

        for (int64_t j = 0; j < nlast; ++j)
        {
            mat[RNG::ctr_type::static_size * nit + j] = rv[j];
        }

        c.incr();
    }

    return RNGState<RNG> {c, k};
}





template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v)
{
    size_t n = v.size();
    os << "{";
    if (n)
    {
        os << v[0];
        for (size_t i = 1; i < n; ++i)
            os << ", " << v[i];
    }
    os << "}";
    return os;
}



template <typename T, typename RNG, typename OP>
auto run_test(int64_t m, int64_t n, T *mat)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    RNGState<RNG> seed;
    fill_rmat<T,RNG,OP>(m, n, mat, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    return (t1 - t0).count();
}


int main(int argc, char **argv)
{
    using T = float;
    using RNG = r123::Philox4x32;
    using OP = boxmul;

    int64_t m = atoi(argv[1]);
    int64_t n = atoi(argv[2]);
    int64_t d = m*n;

    std::vector<T> mat(d);

    auto dt = run_test<T,RNG,OP>(m, n, mat.data());

    std::cerr << "[" << typeid(RNG).name() << ", "
        << typeid(OP).name() << "] dt = " << dt << std::endl;

    if (d < 100)
        std::cerr << "mat = " << mat << std::endl;

    return 0;
}

