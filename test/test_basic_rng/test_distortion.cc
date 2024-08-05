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
using RandBLAS::DenseDistName;
using RandBLAS::RNGState;

#include "rng_common.hh"

#include <lapack.hh>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// function [N_] = get_min_dimension(p_, tau_)
//     N_ = ceil( ((sqrt(2*log(1 ./ p_)) + 1) ./ tau_).^2 );
// end

int64_t get_min_dimension(double p, double tau) {
    double val = std::sqrt(-2 * std::log(p)) + 1;
    val /= tau;
    val *= val;
    return (int64_t) std::ceil(val);
}

// TODO: implement Cholesky and Krylov subspace method for computing condition number.
/*


*/

class TestSubspaceDistortion : public ::testing::Test {
    protected:

    template <typename T>
    void run_gaussian(
        T distortion, T tau, T p_fail_bound, uint32_t key
    ) {
        /**
         *  Generate a d-by-N Gaussian matrix, where d = gamma*N,
         *  gamma = (r/delta)^2, and N is the smallest integer where n > N implies
         *  n*(r - 1  - 1/sqrt(n))^2 >= 2*log(1/p_fail_bound).
         *  One can verify that this value for N is given as
         *      N = ceil( ([sqrt(2*log(1/p)) + 1]/(r-1))^2 )   if   r > 1
         * 
         *  With probability at least 1 - p_fail_bound, the spectrum
         *  of the generated matrix will lay in the interval
         *  [1 - distortion, 1 + distortion].
         * 
         * ----------------------
         * Temporary notes
         * ----------------------
         * Find N = min{ n : exp(-t^2 gamma n ) <= p_fail_bound }, where
         *       t := delta - (gamma)^{-1/2}(1 + 1/sqrt(n)), and 
         *   gamma := (r/delta)^2.
         * 
         * Choosing gamma of this form with r > 1 ensures that no 
         * matter the value of delta in (0, 1) there always an N
         * so that probability bound holds whenever n >= N. We know of no bounds
         * available when r is in (0, 1).
         * 
         * ---------------------
         * if delta = 1/sqrt(2) then we're looking at gamma = 2*r^2.
         * Or, setting r = 1 + tau for tau > 0, we're looking at gamma=2*(1+tau)^2
         * to get convergence rate tau in the sense that we fall below a target
         * failure probability once
         *  N = ceil( ( [sqrt(2*log(1/p)) + 1] / tau )^2 )
         *  
         */ 
        int64_t N = get_min_dimension(p_fail_bound, tau);
        int64_t d = std::ceil( std::pow((1 + tau) / distortion, 2) );
        DenseDist D(d, N, DenseDistName::Gaussian);
        std::vector<T> S(d*N);
        RandBLAS::fill_dense(D, S.data(), {key});
        // need LAPACK ...
        return;
    }
};
