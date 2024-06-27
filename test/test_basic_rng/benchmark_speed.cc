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

#include "RandBLAS/config.h"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense_skops.hh"

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
auto run_test(RandBLAS::DenseDist D, T *mat)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    RNGState<RNG> seed;
    RandBLAS::dense::fill_dense_submat_impl<T,RNG,OP>(D.n_cols, mat, D.n_rows, D.n_cols, 0, seed);
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
    RandBLAS::DenseDist dist{m, n, RandBLAS::DenseDistName::Uniform};

    std::vector<T> mat(d);

    auto dt = run_test<T,RNG,OP>(dist, mat.data());

    std::cerr << "[" << typeid(RNG).name() << ", "
        << typeid(OP).name() << "] dt = " << dt << std::endl;

    if (d < 100)
        std::cerr << "mat = " << mat << std::endl;

    return 0;
}

