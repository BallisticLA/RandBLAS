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

#include "RandBLAS.hh"
#include "RandBLAS/base.hh"
#include "RandBLAS/trig_skops.hh"
#include <blas.hh>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <gtest/gtest.h>

using RandBLAS::trig::lmiget;
using RandBLAS::trig::rmiget;
using RandBLAS::generate_rademacher_vector_r123;
using RandBLAS::apply_diagonal_rademacher;
using RandBLAS::permuteRowsToTop;
using RandBLAS::permuteColsToLeft;
using RandBLAS::fht_dispatch;
using Eigen::MatrixXd;


class TestLMIGET : public::testing::Test
{
    protected:
    virtual void SetUp(){};

    virtual void TearDown(){};

    inline static std::vector<uint32_t> keys {0, 42};

    // Helper function for explicitly generating a Hadamard matrix
    // (Note that ColMajor and RowMajor storage is identical for `H`)
    static std::vector<double> generate_hadamard(int64_t log_n) {
        int64_t size = 1 << log_n;  // size = 2^n
        std::vector<std::vector<double>> H(size, std::vector<double>(size, 1));  // Initialize H_1

        // Sylvester's construction: recursively build the matrix
        for (int n = 1; n <= log_n; ++n) {
            double curr_size = 1 << n;  // Current size of the matrix is 2^n
            for (int i = 0; i < curr_size / 2; ++i) {
                for (int j = 0; j < curr_size / 2; ++j) {
                    // Fill the bottom-left and bottom-right quadrants
                    H[i + curr_size / 2][j] = H[i][j];       // Copy the top-left quadrant to bottom-left
                    H[i][j + curr_size / 2] = H[i][j];       // Copy the top-left quadrant to top-right
                    H[i + curr_size / 2][j + curr_size / 2] = -H[i][j]; // Fill bottom-right with negative values
                }
            }
        }

        // Flatten into a vector in ColMajor order
        std::vector<double> H_flat(size * size);

        for (int col = 0; col < size; ++col) {
            for (int row = 0; row < size; ++row) {
                H_flat[col * size + row] = H[row][col];
            }
        }

        return H_flat;
    }

    static std::vector<double> generate_random_vector(int size, double lower_bound, double upper_bound) {
        // Create a random device and seed the random number generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Define the distribution range for the random doubles
        std::uniform_real_distribution<> dist(lower_bound, upper_bound);

        // Create a vector of the specified size
        std::vector<double> random_vector(size);

        // Generate random doubles and fill the vector
        for (int i = 0; i < size; ++i) {
            random_vector[i] = dist(gen);
        }

        return random_vector;
    }

    enum class transforms {diag_scale, hadamard, permute};

    // Tests to verify correctness of each of the transforms
    template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
    static void correctness(
        uint32_t seed,
        transforms transform,
        int64_t m, // Generated data matrix, `A` is of size `(m x n)`
        int64_t n,
        int64_t d, // The permutation matrix permutes `d` of the final rows/cols
        bool left,
        blas::Layout layout,
        double epsilon=1e-5
    ) {
        // Grabbing a random matrix
        std::vector<double> A_vec = generate_random_vector(m * n, 0.0, 10.0);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_col = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(A_vec.data(), m, n);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_row(A_vec.data(), m, n);

        // Deep copy
        MatrixXd B;
        if(layout == blas::Layout::RowMajor)
        B = A_row;
        else
        B = A_col;

        switch (transform) {
        case transforms::permute: {
            // Simply compares against Eigen::PermutationMatrix
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(5);

            std::vector<int> V = left ? std::vector<int>(m) : std::vector<int>(n);

            int cnt = 0;
            for(int i = 0; i < V.size(); i++) {
                if(i == 0)
                V[i] = V.size() - 1;
                else if(i == V.size() - 1)
                V[i] = 0;
                else
                V[i] = cnt;
                cnt++;
            }

            Eigen::VectorXi indices = Eigen::Map<Eigen::VectorXi>(V.data(), V.size());

            // Set the indices in the permutation matrix
            perm.indices() = indices;

            sint_t* v = new sint_t;
            *v = V.size() - 1;
            if(left) {
                if(layout == blas::Layout::RowMajor)
                RandBLAS::permuteRowsToTop(layout, m, n, v, 1, A_row.data());
                else
                RandBLAS::permuteRowsToTop(layout, m, n, v, 1, A_col.data());
            }
            else {
                if(layout == blas::Layout::RowMajor)
                RandBLAS::permuteColsToLeft(layout, m, n, v, 1, A_row.data());
                else
                RandBLAS::permuteColsToLeft(layout, m, n, v, 1, A_col.data());
            }

            // Or just do A.isApprox(B)
            double norm_permute = 0.0;
            if(left) {
                if(layout == blas::Layout::RowMajor)
                norm_permute = (A_row - perm * B).norm();
                else
                norm_permute = (A_col - perm * B).norm();
            }
            else {
            if(layout == blas::Layout::RowMajor)
                norm_permute = (A_row - B * perm).norm();
            else
                norm_permute = (A_row - B * perm).norm();
            }

            // Or do A.isApprox(H * B)
            randblas_require(norm_permute < epsilon);

            break;
        }
        case transforms::hadamard: {
            // Here, simply check against explicit application of the Hadamard matrix
            int ld = (left) ? m : n;
            if(layout == blas::Layout::ColMajor)
            RandBLAS::fht_dispatch(left, layout, A_col.data(), std::log2(ld), m, n);
            else
            RandBLAS::fht_dispatch(left, layout, A_row.data(), std::log2(ld), m, n);

            std::vector<double> H_vec = generate_hadamard(std::log2(ld));
            //TODO: Should have a check here to enforce that `m` and `n` are powers of 2 (since
            // my `generate_hadamard` function does not take care to pad an input matrix)
            MatrixXd H = Eigen::Map<MatrixXd>(H_vec.data(), int(std::pow(2, std::log2(ld))), int(std::pow(2, std::log2(ld))));

            double norm_hadamard = 0.0;
            if(left)
            if(layout == blas::Layout::RowMajor)
                norm_hadamard = (A_row - H * B).norm();
            else
                norm_hadamard = (A_col - H * B).norm();
            else
            if(layout == blas::Layout::RowMajor)
                norm_hadamard = (A_row - B * H).norm();
            else
                norm_hadamard = (A_col - B * H).norm();

            randblas_require(norm_hadamard < epsilon);

            break;
        }
        case transforms::diag_scale: {
            // Scales all rows/cols by -1 and checks if A == -A
            std::vector<sint_t> buff = left ? std::vector<sint_t>(n, -1) : std::vector<sint_t>(m, -1);

            double norm_diag = 0.0;
            if(layout == blas::Layout::RowMajor) {
            RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_row.data(), buff.data());
            double norm_diag = (A_row + B).norm();
            }
            else {
            RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_col.data(), buff.data());
            double norm_diag = (A_col + B).norm();
            }

            randblas_require(norm_diag < epsilon);

            break;
        }
        }
    }
};

////////////////////////////////////////////////////////////////////////
//
//
//      Checking correctness of each of the transforms
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLMIGET, test_diag_left_colmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::diag_scale,
            100,
            100,
            0,
            true,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_diag_right_colmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::diag_scale,
            100,
            100,
            0,
            false,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_diag_left_rowmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::diag_scale,
            100,
            100,
            0,
            true,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_diag_right_rowmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::diag_scale,
            100,
            100,
            0,
            false,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_permute_left_colmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::permute,
            100,
            100,
            0,
            true,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_permute_right_colmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::permute,
            100,
            100,
            0,
            false,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_permute_left_rowmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::permute,
            100,
            100,
            0,
            true,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_permute_right_rowmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::permute,
            100,
            100,
            0,
            false,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_hadamard_left_colmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::hadamard,
            128,
            100,
            0,
            true,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_hadamard_right_colmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::hadamard,
            100,
            128,
            0,
            false,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_hadamard_left_rowmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::hadamard,
            128,
            100,
            0,
            true,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_hadamard_right_rowmajor) {
    for(uint32_t seed: keys)
        correctness<double>(
            seed,
            transforms::hadamard,
            100,
            128,
            0,
            false,
            blas::Layout::RowMajor
        );
}
