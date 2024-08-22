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
#include "RandBLAS/base.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/dense_skops.hh"
using RandBLAS::RNGState;
using RandBLAS::DenseDistName;
#include "rng_common.hh"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>
#include <gtest/gtest.h>
#include <stdexcept>



class TestScalarDistributions : public ::testing::Test {
    protected:

    // This is really for distributions whose CDFs "F" are continuous and strictly increasing
    // on the interval [a, b] where F(a) = 0 and F(b) = 1.
 
    template <typename T>
    static void kolmogorov_smirnov_tester(
        std::vector<T> &samples, double critical_value, DenseDistName dn
    ) { 
        auto F_true = [dn](T x) {
            if (dn == DenseDistName::Gaussian) {
                return RandBLAS_StatTests::standard_normal_cdf(x);
            } else if (dn == DenseDistName::Uniform) {
                return RandBLAS_StatTests::uniform_syminterval_cdf(x, (T) std::sqrt(3));
            } else {
                std::string msg = "Unrecognized distributions name";
                throw std::runtime_error(msg);
            }
        };
        auto N = (int) samples.size();
        /** 
         *  Let L(x) = |F_empirical(x) - F_true(x)|. The KS test testatistic is
         *
         *      ts = sup_{all x} L(x).
         * 
         *  Now set s = sorted(samples), and partition the real line into
         * 
         *      I_0     = (-infty,  s[0  ]),  ...
         *      I_1     = [s[0  ],  s[1  ]),  ...
         *      I_2     = [s[1  ],  s[2  ]),  ...
         *      I_{N-1} = [s[N-2],  s[N-1]),  ...
         *      I_N     = [s[N-1],  +infty).
         * 
         *  Then, provided F_true is continuous, we have 
         * 
         *      sup{ L(x) : x in I_j } = max{ 
         *              |F_true(inf(I_j)) - j/N|, |F_true(sup(I_j)) - j/N|
         *      }
         * 
         *  for j = 0, ..., N.
         */
        samples.push_back( std::numeric_limits<T>::infinity());
        samples.push_back(-std::numeric_limits<T>::infinity());
        std::sort(samples.begin(), samples.end(), [](T a, T b) {return (a < b);});
        for (int64_t j = 0; j <= N; ++j) {
            T temp1 = F_true(samples[j    ]);
            T temp2 = F_true(samples[j + 1]);
            T empirical = ((T)j)/((T)N);
            T val1 = std::abs(temp1 - empirical);
            T val2 = std::abs(temp2 - empirical);
            T supLx_on_Ij = std::max(val1, val2);
            ASSERT_LE(supLx_on_Ij, critical_value) 
                << "\nj = " << j << " of N = " << N
                << "\nF_true(inf(I_j)) = " << temp1 
                << "\nF_true(sup(I_j)) = " << temp2 
                << "\nI_j = [" << samples[j] << ", " << samples[j+1] << ")";
        }
        return;
    }

    template <typename T>
    static void run(double significance, int64_t num_samples, DenseDistName dn, uint32_t seed) {
        using RandBLAS_StatTests::KolmogorovSmirnovConstants::critical_value_rep_mutator;
        auto critical_value = critical_value_rep_mutator(num_samples, significance);
        RNGState state(seed);
        std::vector<T> samples(num_samples, -1);
        RandBLAS::fill_dense({num_samples, 1, dn, RandBLAS::MajorAxis::Long}, samples.data(), state);
        kolmogorov_smirnov_tester(samples, critical_value, dn);
        return;
    }
};

TEST_F(TestScalarDistributions, uniform_ks_generous) {
    double s = 1e-6;
    for (uint32_t i = 999; i < 1011; ++i) {
        run<double>(s, 100000, DenseDistName::Uniform, i);
        run<double>(s, 10000,  DenseDistName::Uniform, i*i);
        run<double>(s, 1000,   DenseDistName::Uniform, i*i*i);
    }
}

TEST_F(TestScalarDistributions, uniform_ks_moderate) {
    double s = 1e-4;
    run<float>(s, 100000, DenseDistName::Uniform, 0);
    run<float>(s, 10000,  DenseDistName::Uniform, 0);
    run<float>(s, 1000,   DenseDistName::Uniform, 0);
}

TEST_F(TestScalarDistributions, uniform_ks_skeptical) {
    double s = 1e-2;
    run<float>(s, 100000, DenseDistName::Uniform, 0);
    run<float>(s, 10000,  DenseDistName::Uniform, 0);
    run<float>(s, 1000,   DenseDistName::Uniform, 0);
}

TEST_F(TestScalarDistributions, guassian_ks_generous) {
    double s = 1e-6;
    for (uint32_t i = 99; i < 103; ++i) {
        run<double>(s, 100000, DenseDistName::Gaussian, i);
        run<double>(s, 10000,  DenseDistName::Gaussian, i*i);
        run<double>(s, 1000,   DenseDistName::Gaussian, i*i*i);
    }
}

TEST_F(TestScalarDistributions, guassian_ks_moderate) {
    double s = 1e-4;
    run<float>(s, 100000, DenseDistName::Gaussian, 0);
    run<float>(s, 10000,  DenseDistName::Gaussian, 0);
    run<float>(s, 1000,   DenseDistName::Gaussian, 0);
}

TEST_F(TestScalarDistributions, guassian_ks_skeptical) {
    double s = 1e-2;
    run<float>(s, 100000, DenseDistName::Gaussian, 0);
    run<float>(s, 10000,  DenseDistName::Gaussian, 0);
    run<float>(s, 1000,   DenseDistName::Gaussian, 0);
}
