
#include <Random123/boxmuller.hpp>
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/aes.h>
#include <Random123/ars.h>

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





namespace std {
  template<> struct tuple_size<r123::float2> { static constexpr size_t value = 2; };
  template<> struct tuple_element<0, r123::float2> { using type = float; };
  template<> struct tuple_element<1, r123::float2> { using type = float; };
}

namespace r123 {
template<std::size_t I>
std::tuple_element_t<I, r123::float2> get(const r123::float2 &f2)
{
  if constexpr (I == 0) return f2.x;
  if constexpr (I == 1) return f2.y;
}

}

template <typename RNG, typename CTR = typename RNG::ctr_type, typename KEY = typename RNG::key_type, typename UKEY = typename RNG::ukey_type>
struct RNGState
{
    using generator = RNG;

    RNGState() : counter{{}}, key(UKEY{{}}) {}

    RNGState(const CTR &c, const KEY &k) : counter(c), key(k) {}
    RNGState(CTR &&c, KEY &&k) : counter(std::move(c)), key(std::move(k)) {}

    //RNGState(const CTR &c, const UKEY &k) : counter(c), key(k) {}
    //RNGState(CTR &&c, UKEY &&k) : counter(std::move(c)), key(std::move(k)) {}

    auto operator()() { return std::make_tuple(counter, key); }
    auto operator()() const { return std::make_tuple(counter, key); }

    CTR counter;
    KEY key;
};

struct boxmul
{
    template <typename RNG, typename CTR = typename RNG::ctr_type, typename KEY = typename RNG::key_type>
    static
    auto generate(RNG &rng, CTR &&c, KEY &&k)
    {
        auto r = rng(c, k);
        auto [v0, v1] = r123::boxmuller(r.v[0], r.v[1]);
        auto [v2, v3] = r123::boxmuller(r.v[2], r.v[3]);
        return std::array {v0, v1, v2, v3};
    }
};



struct uneg11
{
    template <typename RNG, typename CTR = typename RNG::ctr_type, typename KEY = typename RNG::key_type,
        typename VAL = typename std::conditional
            <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
    static
    auto generate(RNG &rng, CTR &&c, KEY &&k)
    {
        auto r = rng(c, k);
        return r123::uneg11all<VAL>(r);
    }
};


template <typename T, typename RNG, typename OP, typename CTR = typename RNG::ctr_type, typename KEY = typename RNG::key_type>
auto generate(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState<RNG> &seed
) {
    RNG rng;
    auto [c, k] = seed();

    int64_t dim = n_rows * n_cols;
    int64_t nit = dim / 4;
    int64_t nlast = dim % 4;

#if defined(USE_OMP)
    #pragma omp parallel firstprivate(c, k)
    {
        int q = omp_get_thread_num();

        // give each thread a unique key so each will generate a unique
        // sequence
        k[0] = q;

        #pragma omp for
#endif
        for (int64_t i = 0; i < nit; ++i)
        {
            auto v = OP::generate(rng, c, k);

            mat[4*i    ] = v[0];
            mat[4*i + 1] = v[1];
            mat[4*i + 2] = v[2];
            mat[4*i + 3] = v[3];

            c.incr();
        }
#if defined(USE_OMP)
    }

    c.incr(nit);
#endif

    if (nlast)
    {
        auto v = OP::generate(rng, c, k);

        for (int64_t j = 0; j < nlast; ++j)
            mat[4*nit + j] = v[j];

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
    generate<T,RNG,OP>(m, n, mat, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    return (t1 - t0).count();
}


int main(int argc, char **argv)
{
    using RNG = r123::Philox4x32;
    using OP = boxmul; //uneg11;



    //r123::Array4x32::value_type a;
    RNG::ctr_type::value_type a;

    int64_t m = atoi(argv[1]);
    int64_t n = atoi(argv[2]);
    int64_t d = m*n;

    std::vector<float> mat(d);


    auto phi = run_test<float, r123::Philox4x32, OP>(m, n, mat.data());
#if 0
    auto tfr = run_test<float, r123::Threefry4x32, OP>(m, n, mat.data());
    //#if defined(__AES__)
    auto aes = run_test<float, r123::AESNI4x32, OP>(m, n, mat.data());
    auto ars = run_test<float, r123::ARS4x32, OP>(m, n, mat.data());
#endif

    std::cerr << "phi = " << phi << std::endl
#if 0
        << "tfr = " << tfr << std::endl
//#if defined(__AES__)
        << "aes = " << aes << std::endl
        << "ars = " << ars << std::endl
#endif
        ;

    if (d < 100)
        std::cerr << "mat = " << mat << std::endl;

    return 0;
}

