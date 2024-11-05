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
#include "RandBLAS/util.hh"
#include <blas.hh>

#include <cmath>
#include <random>
#include <gtest/gtest.h>

class TestLMIGET : public::testing::Test
{
    protected:
    virtual void SetUp(){};

    virtual void TearDown(){};

    inline static std::vector<uint32_t> keys {42};

    // Helper function for generating a row-permutation matrix
    // in ColMajor format
    //NOTE: Need to deallocate memory explicitly
    template <typename T>
    static T* generate_row_permutation_row_major(int size, const std::vector<int>& final_row_order) {
        // Allocate memory for a size x size matrix and initialize with zeros
        T* perm_matrix = new T[size * size]();

        // Fill in the matrix with '1's at positions corresponding to the final row order
        for (int i = 0; i < size; ++i) {
            int final_row = final_row_order[i];
            perm_matrix[i * size + final_row] = 1.0;  // Set the '1' in the permuted position (RowMajor)
        }

        return perm_matrix;
    }

    // Helper function for explicitly generating a Hadamard matrix
    // (Note that ColMajor and RowMajor storage is identical for `H`)
    template <typename T>
    static std::vector<T> generate_hadamard(int64_t log_n) {
        int64_t size = 1 << log_n;  // size = 2^n
        std::vector<std::vector<T>> H(size, std::vector<T>(size, 1));  // Initialize H_1

        // Sylvester's construction: recursively build the matrix
        for (int n = 1; n <= log_n; ++n) {
            T curr_size = 1 << n;  // Current size of the matrix is 2^n
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
        std::vector<T> H_flat(size * size);

        for (int col = 0; col < size; ++col) {
            for (int row = 0; row < size; ++row) {
                H_flat[col * size + row] = H[row][col];
            }
        }

        return H_flat;
    }

    template <typename T>
    static std::vector<T> generate_random_vector(int size, T lower_bound, T upper_bound) {
        // Create a random device and seed the random number generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Define the distribution range for the random Ts
        std::uniform_real_distribution<> dist(lower_bound, upper_bound);

        // Create a vector of the specified size
        std::vector<T> random_vector(size);

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
        bool left,
        blas::Layout layout,
        T tol=RandBLAS::sqrt_epsilon<T>()
    ) {
        // Grabbing a random matrix
        std::vector<T> A_vec = generate_random_vector(m * n, 0.0, 10.0);
        std::vector<T> B_vec(A_vec);

        switch (transform) {
        case transforms::permute: {
            int ld = (left) ? m : n;

            std::vector<int> V = std::vector<int>(ld);

            int cnt = 0;
            // int cnt = 0;
            for(int i = 0; i < V.size(); i++) {
                if(i == 0)
                    V[i] = V.size() - 1;
                else if(i == V.size() - 1)
                    V[i] = 0;
                else
                    V[i] = cnt;
                cnt++;
            }

            T* perm_matrix = generate_row_permutation_row_major<T>(ld, V);
            T* true_perm_matrix = new T[m * n];

            sint_t* v = new sint_t;
            *v = V.size() - 1;

            if(left) {
                if(layout == blas::Layout::RowMajor) {
                    blas::gemm(
                        blas::Layout::RowMajor,
                        blas::Op::Trans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        m,
                        1.0,
                        perm_matrix,
                        m,
                        A_vec.data(),
                        n,
                        0.0,
                        true_perm_matrix,
                        n
                    );
                    RandBLAS::permute_rows_to_top(layout, m, n, v, 1, A_vec.data());
                }
                else {
                    blas::gemm(
                        blas::Layout::ColMajor,
                        blas::Op::Trans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        m,
                        1.0,
                        perm_matrix,
                        m,
                        A_vec.data(),
                        m,
                        0.0,
                        true_perm_matrix,
                        m
                    );
                    RandBLAS::permute_rows_to_top(layout, m, n, v, 1, A_vec.data());
                }
            }
            else {
                if(layout == blas::Layout::RowMajor) {
                    blas::gemm(
                        blas::Layout::RowMajor,
                        blas::Op::NoTrans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        n,
                        1.0,
                        A_vec.data(),
                        n,
                        perm_matrix,
                        n,
                        0.0,
                        true_perm_matrix,
                        n
                    );
                    RandBLAS::permute_cols_to_left(layout, m, n, v, 1, A_vec.data());
                }
                else {
                    blas::gemm(
                        blas::Layout::ColMajor,
                        blas::Op::NoTrans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        n,
                        1.0,
                        A_vec.data(),
                        m,
                        perm_matrix,
                        n,
                        0.0,
                        true_perm_matrix,
                        m
                    );
                    RandBLAS::permute_cols_to_left(layout, m, n, v, 1, A_vec.data());
                }
            }

            T norm_permute = 0.0;
            blas::axpy(m * n, -1.0, A_vec.data(), 1, true_perm_matrix, 1);
            norm_permute = blas::nrm2(m * n, true_perm_matrix, 1);

            randblas_require(norm_permute < tol);

            delete [] perm_matrix;
            delete [] true_perm_matrix;

            break;
        }
        case transforms::hadamard: {
            // Here, simply check against explicit application of the Hadamard matrix

            int ld = (left) ? m : n;
            RandBLAS::fht_dispatch(left, layout, m, n, std::log2(ld), A_vec.data());

            std::vector<T> H_vec = generate_hadamard<double>(std::log2(ld));
            //TODO: Should have a check here to enforce that `m` and `n` are powers of 2 (since
            // my `generate_hadamard` function does not take care to pad an input matrix)

            T* true_H_mult = new T[m * n];

            T norm_hadamard = 0.0;

            if(left) {
                if(layout == blas::Layout::RowMajor) {
                    blas::gemm(
                        blas::Layout::RowMajor,
                        blas::Op::NoTrans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        m,
                        1.0,
                        H_vec.data(),
                        m,
                        B_vec.data(),
                        n,
                        0.0,
                        true_H_mult,
                        n
                    );
                    blas::axpy(m * n, -1.0, A_vec.data(), 1, true_H_mult, 1);
                    norm_hadamard = blas::nrm2(m * n, true_H_mult, 1);
                }
                else {
                    blas::gemm(
                        blas::Layout::ColMajor,
                        blas::Op::NoTrans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        m,
                        1.0,
                        H_vec.data(),
                        m,
                        B_vec.data(),
                        m,
                        0.0,
                        true_H_mult,
                        m
                    );
                    blas::axpy(m * n, -1.0, A_vec.data(), 1, true_H_mult, 1);
                    norm_hadamard = blas::nrm2(m * n, true_H_mult, 1);
                }
            }
            else {
                if(layout == blas::Layout::RowMajor) {
                    blas::gemm(
                        blas::Layout::RowMajor,
                        blas::Op::NoTrans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        n,
                        1.0,
                        B_vec.data(),
                        n,
                        H_vec.data(),
                        n,
                        0.0,
                        true_H_mult,
                        n
                    );
                    blas::axpy(m * n, -1.0, A_vec.data(), 1, true_H_mult, 1);
                    norm_hadamard = blas::nrm2(m * n, true_H_mult, 1);
                }
                else {
                    blas::gemm(
                        blas::Layout::ColMajor,
                        blas::Op::NoTrans,
                        blas::Op::NoTrans,
                        m,
                        n,
                        n,
                        1.0,
                        B_vec.data(),
                        m,
                        H_vec.data(),
                        n,
                        0.0,
                        true_H_mult,
                        m
                    );
                    blas::axpy(m * n, -1.0, A_vec.data(), 1, true_H_mult, 1);
                    norm_hadamard = blas::nrm2(m * n, true_H_mult, 1);
                }
            }

            randblas_require(norm_hadamard < tol);

            delete [] true_H_mult;

            break;
        }
        case transforms::diag_scale: {
            // Scales all rows/cols by -1 and checks if A == -A
            std::vector<sint_t> buff = left ? std::vector<sint_t>(n, -1) : std::vector<sint_t>(m, -1);

            RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_vec.data(), buff.data());

            T norm_diag = 0.0;
            blas::axpy(m * n, 1.0, A_vec.data(), 1, B_vec.data(), 1);
            norm_diag = blas::nrm2(m * n, B_vec.data(), 1);

            randblas_require(norm_diag < tol);

            break;
        }
        }
    }

    template <typename T, typename RNG = RandBLAS::DefaultRNG, RandBLAS::SignedInteger sint_t = int64_t>
    static void drivers_inverse(
        uint32_t seed,
        int64_t m,
        int64_t n,
        int64_t d,
        int64_t left,
        blas::Layout layout,
        T tol=RandBLAS::sqrt_epsilon<T>()
    ) {
        // Grabbing a random matrix
        std::vector<T> A_vec = generate_random_vector(m * n, 0.0, 10.0);
        std::vector<T> B_vec(A_vec);

        RandBLAS::RNGState<RNG> seed_state(seed);

        // Aggregating information about the matrix
        RandBLAS::trig::HadamardMixingOp<> hmo(
            left,
            layout,
            m,
            n,
            d
        );

        // Performing the transform and...
        RandBLAS::trig::miget(hmo, seed_state, A_vec.data());
        RandBLAS::trig::invert_hadamard(hmo, A_vec.data());
        // ... inverting

        T norm_inverse = 0.0;

        blas::axpy(m * n, -1.0, A_vec.data(), 1, B_vec.data(), 1);
        norm_inverse = blas::nrm2(m * n, B_vec.data(), 1);

        randblas_require(norm_inverse < tol);
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
            false,
            blas::Layout::RowMajor
        );
}

////////////////////////////////////////////////////////////////////////
//
//
//      Verifying correctness (in invertibility) of user-facing functions
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLMIGET, test_user_inverse_left_colmajor) {
    for(uint32_t seed: keys)
        drivers_inverse<double>(
            seed,
            128,
            128,
            25,
            true,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_user_inverse_right_colmajor) {
    for(uint32_t seed: keys)
        drivers_inverse<double>(
            seed,
            128,
            128,
            25,
            false,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_user_inverse_left_rowmajor) {
    for(uint32_t seed: keys)
        drivers_inverse<double>(
            seed,
            128,
            128,
            25,
            true,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_user_inverse_right_rowmajor) {
    for(uint32_t seed: keys)
        drivers_inverse<double>(
            seed,
            128,
            128,
            25,
            false,
            blas::Layout::RowMajor
        );
}
