
#include "RandBLAS/config.h"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"

#include <iostream>
#include <vector>
#include <typeinfo>
#include <cstring>
#include <chrono>

using namespace RandBLAS;


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
    base::RNGState<RNG> seed;
    dense::fill_rmat<T,RNG,OP>(m, n, mat, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    return (t1 - t0).count();
}


int main(int argc, char **argv)
{
    (void) argc;

    using T = float;
    using RNG = r123::Philox4x32;
    using OP = base::uneg11;

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

