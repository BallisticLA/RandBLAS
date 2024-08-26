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
using RandBLAS::DenseDist;
using RandBLAS::ScalarDist;
using RandBLAS::RNGState;

#include "rng_common.hh"
#include "../handrolled_lapack.hh"

#include <iostream>
#include <vector>
#include <gtest/gtest.h>


class TestSubspaceDistortion : public ::testing::Test {
    protected:

    template <typename T>
    void run_general(ScalarDist name, T distortion, int64_t d, int64_t N, uint32_t key) {
        auto layout = blas::Layout::ColMajor;
        DenseDist D(d, N, name);
        std::vector<T> S(d*N);
        std::cout << "(d, N) = ( " << d << ", " << N << " )\n";
        RandBLAS::RNGState<r123::Philox4x32> state(key);
        auto next_state = RandBLAS::fill_dense(D, S.data(), state);
        T inv_stddev = (name == ScalarDist::Gaussian) ? (T) 1.0 : (T) 1.0;
        blas::scal(d*N, inv_stddev / std::sqrt(d), S.data(), 1);
        std::vector<T> G(N*N, 0.0);
        blas::syrk(layout, blas::Uplo::Upper, blas::Op::Trans, N, d, (T)1.0, S.data(), d, (T)0.0, G.data(), N);
        RandBLAS::symmetrize(layout, blas::Uplo::Upper, N, G.data(), N);
        
        std::vector<T> eigvecs(2*N, 0.0);
        std::vector<T> subwork{};
        T powermethod_reltol = 1e-2;
        T powermethod_failprob = 1e-6;
        auto [lambda_max, lambda_min, ignore] = hr_lapack::exeigs_powermethod(
            N, G.data(), eigvecs.data(), powermethod_reltol, powermethod_failprob, state, subwork
        );
        T sigma_max = std::sqrt(lambda_max);
        T sigma_min = std::sqrt(lambda_min);
        ASSERT_LE(sigma_max, 1+distortion);
        ASSERT_GE(sigma_min, 1-distortion);
        return;
    }

    template <typename T>
    void run_gaussian(T distortion, T tau, T p_fail_bound, uint32_t key) {
        /**
         *  Generate a d-by-N random matrix, where d = gamma*N,
         *  gamma = ((1 + tau)/delta)^2, and N is the smallest integer where n > N implies
         *  n*(tau  - 1/sqrt(n))^2 >= 2*log(1/p_fail_bound).
         *  One can verify that this value for N is given as
         *      N = ceil( ([sqrt(2*log(1/p)) + 1]/ tau )^2 )
         * 
         * ----------------------
         * Temporary notes
         * ----------------------
         * Find N = min{ n : exp(-t^2 gamma n ) <= p_fail_bound }, where
         *       t := delta - (gamma)^{-1/2}(1 + 1/sqrt(n)), and 
         *   gamma := ((1+tau)/delta)^2.
         * 
         * Choosing gamma of this form with tau > 0 ensures that no 
         * matter the value of delta in (0, 1) there always an N
         * so that probability bound holds whenever n >= N.
         * 
         */ 
        double val = std::sqrt(-2 * std::log(p_fail_bound)) + 1;
        val /= tau;
        val *= val;
        int64_t N = (int64_t) std::ceil(val);
        int64_t d = std::ceil( std::pow((1 + tau) / distortion, 2) * N );
        run_general<T>(ScalarDist::Gaussian, distortion, d, N, key);
        return;
    }

    template <typename T>
    void run_uniform(T distortion, T rate, T p_fail_bound, uint32_t key) {
        int64_t N = std::ceil(std::log((T)2 / p_fail_bound) / rate);
        T c6 = 1.0; // definitely not high enough.
        T epsnet_spectralnorm_factor = 1.0; // should be 4.0
        T theta = epsnet_spectralnorm_factor * c6 * (rate + std::log(9));
        int64_t d = std::ceil(N * theta * std::pow(distortion, -2));
        run_general<T>(ScalarDist::Uniform, distortion, d, N, key);
        return;
    }
};

TEST_F(TestSubspaceDistortion, gaussian_rate_100_fail_0001) {
    uint32_t key = 8673309;
    float p_fail = 1e-3;
    for (uint32_t i = 0; i < 3; ++i ) {
        run_gaussian<float>(0.50f, 1.0f, p_fail, key + i);
        run_gaussian<float>(0.25f, 1.0f, p_fail, key + i);
        run_gaussian<float>(0.10f, 1.0f, p_fail, key + i);
    }
}

TEST_F(TestSubspaceDistortion, gaussian_rate_004_fail_0001) {
    uint32_t key = 8673309;
    float p_fail = 1e-3;
    float tau = 0.2f; // the convergence rate depends on tau^2.
    for (uint32_t i = 0; i < 3; ++i ) {
        run_gaussian<float>(0.75f, tau, p_fail, key + i);
        run_gaussian<float>(0.50f, tau, p_fail, key + i);
        run_gaussian<float>(0.25f, tau, p_fail, key + i);
    }
}

TEST_F(TestSubspaceDistortion, uniform_rate_100_fail_0001) {
    uint32_t key = 8673309;
    float p_fail = 1e-3;
    for (uint32_t i = 0; i < 3; ++i ) {
        run_uniform<float>(0.50f, 1.0f, p_fail, key + i);
        run_uniform<float>(0.25f, 1.0f, p_fail, key + i);
        run_uniform<float>(0.10f, 1.0f, p_fail, key + i);
    }
}

TEST_F(TestSubspaceDistortion, uniform_rate_004_fail_0001) {
    uint32_t key = 8673309;
    float p_fail = 1e-3;
    float rate = 0.04;
    for (uint32_t i = 0; i < 3; ++i ) {
        run_uniform<float>(0.50f, rate, p_fail, key + i);
        run_uniform<float>(0.25f, rate, p_fail, key + i);
        run_uniform<float>(0.10f, rate, p_fail, key + i);
    }
}
