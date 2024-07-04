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
using RandBLAS::RNGState;
#include "rng_common.hh"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>
#include <gtest/gtest.h>


class TestSampleIndices : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    static void test_iid_uniform_smoke(int64_t N, int64_t k, uint32_t seed) { 
        RNGState state(seed);
        std::vector<int64_t> samples(k, -1);
        RandBLAS::util::sample_indices_iid_uniform(N, samples.data(), k, state);
        int64_t* data = samples.data();
        for (int64_t i = 0; i < k; ++i) {
            ASSERT_LT(data[i], N);
            ASSERT_GE(data[i], 0);
        }
        return;
    }

    static void index_set_kolmogorov_smirnov_tester(
        std::vector<int64_t> &samples, std::vector<float> &true_cdf, double critical_value
    ) {
        auto num_samples = (int) samples.size();
        auto N = (int64_t) true_cdf.size();
        std::vector<float> sample_cdf(N, 0.0);
        for (int64_t s : samples)
            sample_cdf[s] += 1;
        RandBLAS::util::weights_to_cdf(N, sample_cdf.data());

        for (int i = 0; i < num_samples; ++i) {
            auto diff = (double) std::abs(sample_cdf[i] - true_cdf[i]);
            ASSERT_LT(diff, critical_value);
        }
        return;
    }

    static void test_iid_uniform_kolmogorov_smirnov(int64_t N, double significance, int64_t num_samples, uint32_t seed) {
        using RandBLAS_StatTests::KolmogorovSmirnovConstants::critical_value_rep_mutator;
        auto critical_value = critical_value_rep_mutator(num_samples, significance);

        std::vector<float> true_cdf(N, 1.0);
        RandBLAS::util::weights_to_cdf(N, true_cdf.data());

        RNGState state(seed);
        std::vector<int64_t> samples(num_samples, -1);
        RandBLAS::util::sample_indices_iid_uniform(N, samples.data(), num_samples, state);

        index_set_kolmogorov_smirnov_tester(samples, true_cdf, critical_value);
        return;
    }

    static void test_iid_kolmogorov_smirnov(int64_t N, double significance, int64_t num_samples, uint32_t seed) {
        using RandBLAS_StatTests::KolmogorovSmirnovConstants::critical_value_rep_mutator;
        auto critical_value = critical_value_rep_mutator(num_samples, significance);

        // Make the true CDF 
        std::vector<float> true_cdf{};
        for (int i = 0; i < N; ++i)
            true_cdf.push_back(1.0/((float)i + 1.0));
        RandBLAS::util::weights_to_cdf(N, true_cdf.data());

        RNGState state(seed);
        std::vector<int64_t> samples(num_samples, -1);
        RandBLAS::util::sample_indices_iid(N, true_cdf.data(), num_samples, samples.data(), state);

        index_set_kolmogorov_smirnov_tester(samples, true_cdf, critical_value);
        return;
    }
    
};


TEST_F(TestSampleIndices, smoke_3_x_10) {
    for (uint32_t i = 0; i < 10; ++i)
        test_iid_uniform_smoke(3, 10, i);
}

TEST_F(TestSampleIndices, smoke_10_x_3) {
    for (uint32_t i = 0; i < 10; ++i)
        test_iid_uniform_smoke(10, 3, i);
}

TEST_F(TestSampleIndices, smoke_med) {
    for (uint32_t i = 0; i < 10; ++i)
        test_iid_uniform_smoke((int) 1e6 , 6000, i);
}

TEST_F(TestSampleIndices, smoke_big) {
    int64_t huge_N = std::numeric_limits<int64_t>::max() / 2;
    for (uint32_t i = 0; i < 10; ++i)
        test_iid_uniform_smoke(huge_N, 1000, i);
}

TEST_F(TestSampleIndices, iid_uniform_ks_generous) {
    double s = 1e-6;
    test_iid_uniform_kolmogorov_smirnov(100,     s, 100000, 0);
    test_iid_uniform_kolmogorov_smirnov(10000,   s, 1000,   0);
    test_iid_uniform_kolmogorov_smirnov(1000000, s, 1000,   0);
}

TEST_F(TestSampleIndices, iid_uniform_ks_moderate) {
    float s = 1e-4;
    test_iid_uniform_kolmogorov_smirnov(100,     s, 100000, 0);
    test_iid_uniform_kolmogorov_smirnov(10000,   s, 1000,   0);
    test_iid_uniform_kolmogorov_smirnov(1000000, s, 1000,   0);
}

TEST_F(TestSampleIndices, iid_uniform_ks_skeptical) {
    float s = 1e-2;
    test_iid_uniform_kolmogorov_smirnov(100,     s, 100000, 0);
    test_iid_uniform_kolmogorov_smirnov(10000,   s, 1000,   0);
    test_iid_uniform_kolmogorov_smirnov(1000000, s, 1000,   0);
}


TEST_F(TestSampleIndices, iid_ks_generous) {
    double s = 1e-6;
    test_iid_kolmogorov_smirnov(100,     s, 100000, 0);
    test_iid_kolmogorov_smirnov(10000,   s, 1000,   0);
    test_iid_kolmogorov_smirnov(1000000, s, 1000,   0);
}

TEST_F(TestSampleIndices, iid_ks_moderate) {
    float s = 1e-4;
    test_iid_kolmogorov_smirnov(100,     s, 100000, 0);
    test_iid_kolmogorov_smirnov(10000,   s, 1000,   0);
    test_iid_kolmogorov_smirnov(1000000, s, 1000,   0);
}

TEST_F(TestSampleIndices, iid_ks_skeptical) {
    float s = 1e-2;
    test_iid_kolmogorov_smirnov(100,     s, 100000, 0);
    test_iid_kolmogorov_smirnov(10000,   s, 1000,   0);
    test_iid_kolmogorov_smirnov(1000000, s, 1000,   0);
}



// class TestSampleIndices : public ::testing::Test
// {
//     protected:
    
//     virtual void SetUp(){};

//     virtual void TearDown(){};

//     template<typename T>
//     static void test_basic(
        
//     ) { 
//         return;
//     }
    
// };


// TEST_F(TestSampleIndices, smoke)
// {
//     // do something
// }



