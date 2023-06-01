
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
auto run_test(RandBLAS::dense::DenseDist D, T *mat)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    base::RNGState<RNG> seed;
    RandBLAS::dense::fill_rsubmat_omp<T,RNG,OP>(D.n_cols, mat, D.n_rows, D.n_cols, 0, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    return (t1 - t0).count();
}


int main(int argc, char **argv)
{
    (void) argc;

    using T = float;
    using RNG = r123::Philox4x32;
    using OP = r123ext::uneg11;

    int64_t m = atoi(argv[1]);
    int64_t n = atoi(argv[2]);
    int64_t d = m*n;
    RandBLAS::dense::DenseDist dist{m, n, RandBLAS::dense::DenseDistName::Uniform};

    std::vector<T> mat(d);

    auto dt = run_test<T,RNG,OP>(dist, mat.data());

    std::cerr << "[" << typeid(RNG).name() << ", "
        << typeid(OP).name() << "] dt = " << dt << std::endl;

    if (d < 100)
        std::cerr << "mat = " << mat << std::endl;

    return 0;
}

