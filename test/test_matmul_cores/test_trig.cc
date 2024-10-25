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

// #include "RandBLAS.hh"
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
using RandBLAS::fht_dispatch;
using Eigen::MatrixXd;


class TestLMIGET : public::testing::Test
{
    protected:
    virtual void SetUp(){};

    virtual void TearDown(){};

    inline static std::vector<uint32_t> keys {42};

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
        T epsilon=RandBLAS::sqrt_epsilon<T>()
    ) {
        // Grabbing a random matrix
        std::vector<T> A_vec = generate_random_vector(m * n, 0.0, 10.0);
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

            Eigen::VectorXi indices = Eigen::Map<Eigen::VectorXi>(V.data(), V.size());

            // Set the indices in the permutation matrix
            perm.indices() = indices;

            sint_t* v = new sint_t;
            *v = V.size() - 1;

            if(left) {
                if(layout == blas::Layout::RowMajor)
                    RandBLAS::permute_rows_to_top(layout, m, n, v, 1, A_row.data());
                else
                    RandBLAS::permute_rows_to_top(layout, m, n, v, 1, A_col.data());
            }
            else {
                if(layout == blas::Layout::RowMajor)
                    RandBLAS::permute_cols_to_left(layout, m, n, v, 1, A_row.data());
                else
                    RandBLAS::permute_cols_to_left(layout, m, n, v, 1, A_col.data());
            }

            // Or just do A.isApprox(B)
            T norm_permute = 0.0;
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
                norm_permute = (A_col - B * perm).norm();
            }

            // Or do A.isApprox(H * B)
            randblas_require(norm_permute < epsilon);

            break;
        }
        case transforms::hadamard: {
            // Here, simply check against explicit application of the Hadamard matrix

            int ld = (left) ? m : n;
            if(layout == blas::Layout::ColMajor)
                RandBLAS::fht_dispatch(left, layout, m, n, std::log2(ld), A_col.data());
            else
                RandBLAS::fht_dispatch(left, layout, m, n, std::log2(ld), A_row.data());

            std::vector<T> H_vec = generate_hadamard<double>(std::log2(ld));
            //TODO: Should have a check here to enforce that `m` and `n` are powers of 2 (since
            // my `generate_hadamard` function does not take care to pad an input matrix)
            MatrixXd H = Eigen::Map<MatrixXd>(H_vec.data(), int(std::pow(2, std::log2(ld))), int(std::pow(2, std::log2(ld))));

            T norm_hadamard = 0.0;
            if(left) {
            if(layout == blas::Layout::RowMajor)
                norm_hadamard = (A_row - H * B).norm();
            else
                norm_hadamard = (A_col - H * B).norm();
            }
            else {
            if(layout == blas::Layout::RowMajor)
                norm_hadamard = (A_row - B * H).norm();
            else
                norm_hadamard = (A_col - B * H).norm();
            }

            randblas_require(norm_hadamard < epsilon);

            break;
        }
        case transforms::diag_scale: {
            // Scales all rows/cols by -1 and checks if A == -A
            std::vector<sint_t> buff = left ? std::vector<sint_t>(n, -1) : std::vector<sint_t>(m, -1);

            T norm_diag = 0.0;
            if(layout == blas::Layout::RowMajor) {
            RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_row.data(), buff.data());
            norm_diag = (A_row + B).norm();
            }
            else {
            RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_col.data(), buff.data());
            norm_diag = (A_col + B).norm();
            }

            randblas_require(norm_diag < epsilon);

            break;
        }
        }
    }

    template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
    static void inverse_transform(
        uint32_t seed,
        int64_t m, // Generated data matrix, `A` is of size `(m x n)`
        int64_t n,
        int64_t d,
        bool left,
        blas::Layout layout,
        T epsilon=RandBLAS::sqrt_epsilon<T>()
    ) {
        // Grabbing a random matrix
        std::vector<T> A_vec = generate_random_vector(m * n, 0.0, 10.0);
        // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_col(m, n);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_col = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(A_vec.data(), m, n);
        // A_col.setRandom();
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_row(A_vec.data(), m, n);

        // Deep copy
        MatrixXd B;
        if(layout == blas::Layout::RowMajor)
        B = A_row;
        else
        B = A_col;

        //// Performing \Pi H D
        // Step 1: setup the diagonal scaling
        std::vector<sint_t> buff = left ? std::vector<sint_t>(m, -1) : std::vector<sint_t>(n, -1);

        if(layout == blas::Layout::RowMajor) {
        RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_row.data(), buff.data());
        }
        else {
        RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_col.data(), buff.data());
        }

        // Step 2: apply the hadamard transform
        int ld = (left) ? m : n;
        if(layout == blas::Layout::ColMajor){
            RandBLAS::fht_dispatch(left, layout, m, n, int(std::log2(ld)), A_col.data());
        }
        else {
            RandBLAS::fht_dispatch(left, layout, m, n, int(std::log2(ld)), A_row.data());
        }

        // Step 3: Permuting
        std::vector<int64_t> indices(d);

        std::iota(indices.begin(), indices.end(), 1);
        if(left) {
            if(layout == blas::Layout::RowMajor)
            RandBLAS::permute_rows_to_top(layout, m, n, indices.data(), d, A_row.data());
            else
            RandBLAS::permute_rows_to_top(layout, m, n, indices.data(), d, A_col.data());
        }
        else {
            if(layout == blas::Layout::RowMajor)
            RandBLAS::permute_cols_to_left(layout, m, n, indices.data(), d, A_row.data());
            else
            RandBLAS::permute_cols_to_left(layout, m, n, indices.data(), d, A_col.data());
        }

        //// Performing D H \Pi

        //Step 1: Un-permute
        std::reverse(indices.begin(), indices.end());

        if(left) {
            if(layout == blas::Layout::RowMajor)
            RandBLAS::permute_rows_to_top(layout, m, n, indices.data(), d, A_row.data());
            else
            RandBLAS::permute_rows_to_top(layout, m, n, indices.data(), d, A_col.data());
        }
        else {
            if(layout == blas::Layout::RowMajor)
            RandBLAS::permute_cols_to_left(layout, m, n, indices.data(), d, A_row.data());
            else
            RandBLAS::permute_cols_to_left(layout, m, n, indices.data(), d, A_col.data());
        }

        // Step-2: Apply H^{-1}
        if(layout == blas::Layout::ColMajor) {
            RandBLAS::fht_dispatch(left, layout, m, n, int(std::log2(ld)), A_col.data());
            A_col = A_col * 1/std::pow(2, int(std::log2(ld)));
        }
        else {
            RandBLAS::fht_dispatch(left, layout, m, n, int(std::log2(ld)), A_row.data());
            A_row = A_row * 1/std::pow(2, int(std::log2(ld)));
        }

        //Step-3: Inverting `D`
        if(layout == blas::Layout::RowMajor) {
        RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_row.data(), buff.data());
        }
        else {
        RandBLAS::apply_diagonal_rademacher(left, layout, m, n, A_col.data(), buff.data());
        }

        T norm_inverse = 0.0;

        if(layout == blas::Layout::RowMajor) {
        norm_inverse = (A_row - B).norm();
        }
        else {
        norm_inverse = (A_col - B).norm();
        }

        randblas_require(norm_inverse < epsilon);


    }

    template <typename T, typename RNG = RandBLAS::DefaultRNG, RandBLAS::SignedInteger sint_t = int64_t>
    static void drivers_inverse(
        uint32_t seed,
        int64_t m,
        int64_t n,
        int64_t d,
        int64_t left,
        blas::Layout layout,
        T epsilon=RandBLAS::sqrt_epsilon<T>()
    ) {
        // Grabbing a random matrix
        std::vector<T> A_vec = generate_random_vector(m * n, 0.0, 10.0);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_col = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(A_vec.data(), m, n);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_row(A_vec.data(), m, n);
        RandBLAS::RNGState<RNG> seed_state(seed);

        // Deep copy
        MatrixXd B;
        if(layout == blas::Layout::RowMajor)
            B = A_row;
        else
            B = A_col;

        // Aggregating information about the matrix
        RandBLAS::trig::HadamardMixingOp<> hmo(
            left,
            layout,
            m,
            n,
            d
        );

        // Sketching the matrix
        if(layout == blas::Layout::RowMajor) {
            RandBLAS::trig::miget(hmo, seed_state, A_row.data());
            RandBLAS::trig::invert(hmo, A_row.data());
        }
        else {
            // std::cout << A_col << std::endl;
            // std::cout << "<-------x------->" << std::endl;
            RandBLAS::trig::miget(hmo, seed_state, A_col.data());
            // for(int i = 0; i < d; i ++)
            //     std::cout << hmo.selected_idxs[i] << std::endl;
            // std::cout << A_col << std::endl;
            // std::cout << "<-------x------->" << std::endl;
            RandBLAS::trig::invert(hmo, A_col.data());
            // std::cout << A_col << std::endl;
            // std::cout << "<-------x------->" << std::endl;
        }

        T norm_inverse = 0.0;

        if(layout == blas::Layout::RowMajor) {
            norm_inverse = (A_row - B).norm();
        }
        else {
            norm_inverse = (A_col - B).norm();
        }

        randblas_require(norm_inverse < epsilon);
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
//      Verifying invertibility of the transform
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLMIGET, test_inverse_left_colmajor) {
    for(uint32_t seed: keys)
        inverse_transform<double>(
            seed,
            128,
            100,
            25,
            true,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_inverse_right_colmajor) {
    for(uint32_t seed: keys)
        inverse_transform<double>(
            seed,
            100,
            128,
            25,
            false,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLMIGET, test_inverse_left_rowmajor) {
    for(uint32_t seed: keys)
        inverse_transform<double>(
            seed,
            128,
            100,
            25,
            true,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLMIGET, test_inverse_right_rowmajor) {
    for(uint32_t seed: keys)
        inverse_transform<double>(
            seed,
            100,
            128,
            25,
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
