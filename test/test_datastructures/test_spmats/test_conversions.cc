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

#include "../../comparison.hh"
#include "common.hh"
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace test::test_datastructures::test_spmats;
using namespace RandBLAS::sparse_data::conversions;
using namespace RandBLAS::sparse_data::csc;
using namespace RandBLAS::sparse_data::csr;
using namespace RandBLAS::sparse_data::coo;
using blas::Layout;

class TestSparseTranspose : public ::testing::Test 
{
    protected:
    virtual void SetUp(){};
    virtual void TearDown(){};

    template <typename T = double>
    void test_transposed_csr_as_csc(int64_t n, int64_t m, T p) {
        Layout layout = Layout::ColMajor;
        std::vector<T> A_dense(m * n);
        std::vector<T> At_dense(n * m);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(m, n, layout, A_dense.data(), p, s);
        CSRMatrix<T> A_csr(m, n);
        dense_to_csr(Layout::ColMajor, A_dense.data(), 0.0, A_csr);
        
        CSCMatrix<T> At_csc_view = transpose_as_csc(A_csr, true);
        csc_to_dense(At_csc_view, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        CSCMatrix<T> At_csc_copy = transpose_as_csc(A_csr, false);
        blas::scal(m * n, 0.0, At_dense.data(), 1);
        blas::scal(A_csr.nnz, 0.0, A_csr.vals, 1);
        csc_to_dense(At_csc_copy, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T = double>
    void test_transposed_csc_as_csr(int64_t n, int64_t m, T p) {
        Layout layout = Layout::ColMajor;
        std::vector<T> A_dense(m * n);
        std::vector<T> At_dense(n * m);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(m, n, layout, A_dense.data(), p, s);
        CSCMatrix<T> A_csc(m, n);
        dense_to_csc(Layout::ColMajor, A_dense.data(), 0.0, A_csc);
        
        CSRMatrix<T> At_csr_view = transpose_as_csr(A_csc, true);
        csr_to_dense(At_csr_view, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        CSRMatrix<T> At_csr_copy = transpose_as_csr(A_csc, false);
        blas::scal(m * n, 0.0, At_dense.data(), 1);
        blas::scal(A_csc.nnz, 0.0, A_csc.vals, 1);
        csr_to_dense(At_csr_copy, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }
};

TEST_F(TestSparseTranspose, CSR_TO_CSC_7x20) {
    test_transposed_csr_as_csc(7, 20, 0.05);
    test_transposed_csr_as_csc(7, 20, 0.90);
}

TEST_F(TestSparseTranspose, CSR_TO_CSC_13x5) {
    test_transposed_csr_as_csc(13, 5, 0.05);
    test_transposed_csr_as_csc(13, 5, 0.90);
}

TEST_F(TestSparseTranspose, CSC_TO_CSR_7x20) {
    test_transposed_csc_as_csr(7, 20, 0.05);
    test_transposed_csc_as_csr(7, 20, 0.90);
}

TEST_F(TestSparseTranspose, CSC_TO_CSR_13x5) {
    test_transposed_csc_as_csr(13, 5, 0.05);
    test_transposed_csc_as_csr(13, 5, 0.90);
}