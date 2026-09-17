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

#include "RandBLAS/sksy.hh"
#include <gtest/gtest.h>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

template <typename T, typename I>
void check_symmetric_locality() {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    for (int64_t n : {0, 1, 17, 65, 128, 129})
    for (int64_t d : {0, 1, 7})
    for (auto layout : {blas::Layout::ColMajor, blas::Layout::RowMajor})
    for (auto uplo : {blas::Uplo::Upper, blas::Uplo::Lower})
    for (bool left : {false, true})
    for (int setting : {0, 1, 2, 3}) {
        SCOPED_TRACE(::testing::Message() << n << ',' << d << ',' << int(layout)
                     << ',' << int(uplo) << ',' << left << ',' << setting);
        const T alpha = setting < 2 ? T(0) : T(1.25);
        const T beta = setting % 2 == 0 ? T(0) : T(-.3);
        const int64_t m = left ? d : n, k = left ? n : d;
        const int64_t lda = n + 3, ldb = (layout == blas::Layout::ColMajor ? m : k) + 5;
        auto ix = [layout](int64_t i, int64_t j, int64_t ld) {
            return layout == blas::Layout::ColMajor ? i + j * ld : i * ld + j;
        };
        auto value = [](int64_t i, int64_t j) {
            return T(1. / (1 + std::abs(i - j)) + (i == j ? 2. : 0.));
        };
        std::vector<T> A(lda * n, nan);
        std::vector<T> B(ldb * (layout == blas::Layout::ColMajor ? k : m), T(73));
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                if (uplo == blas::Uplo::Upper ? i <= j : i >= j) {
                    A[ix(i, j, lda)] = value(i, j);
                }
            }
        }
        for (int64_t j = 0; j < k; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                B[ix(i, j, ldb)] = beta == T(0) ? nan : T(.4);
            }
        }
        const int64_t ro = 1, co = 2, sr = left ? d : n, sc = left ? n : d;
        RandBLAS::sparse_data::COOMatrix<T, I> S(sr + 3, sc + 4);
        const int64_t nnz = 3 * (std::max(sr, sc) + 3);
        S.reserve(nnz);
        for (int64_t p = 0; p < nnz; ++p) {
            // Includes duplicate entries and entries outside the window.
            S.rows[p] = (p / 2) % (sr + 3);
            S.cols[p] = (p / 3) % (sc + 4);
            S.vals[p] = T((p % 5) - 2) / T(3);
        }
        auto reference = B;
        for (int64_t j = 0; j < k; ++j) for (int64_t i = 0; i < m; ++i) {
            long double sum = 0;
            if (alpha != T(0)) {
                for (int64_t p = 0; p < nnz; ++p) {
                    const int64_t r = S.rows[p] - ro, c = S.cols[p] - co;
                    if (r < 0 || r >= sr || c < 0 || c >= sc) continue;
                    if (left && r == i) sum += static_cast<long double>(S.vals[p]) * value(c, j);
                    if (!left && c == j) sum += static_cast<long double>(S.vals[p]) * value(i, r);
                }
            }
            reference[ix(i, j, ldb)] = alpha * sum + (beta == T(0) ? T(0) : beta * T(.4));
        }
        std::vector<T> one;
        for ([[maybe_unused]] int threads : {1, 4}) {
#if defined(RandBLAS_HAS_OpenMP)
            omp_set_num_threads(threads);
#endif
            auto actual = B;
            RandBLAS::util::lascl(layout, m, k, beta, actual.data(), ldb);
            if (left) {
                RandBLAS::sparse_data::coo_lsksys(layout, uplo, d, n, alpha, S, ro, co,
                    alpha == T(0) ? nullptr : A.data(), lda, actual.data(), ldb);
            } else {
                RandBLAS::sparse_data::coo_rsksys(layout, uplo, n, d, alpha,
                    alpha == T(0) ? nullptr : A.data(), lda, S, ro, co, actual.data(), ldb);
            }
            for (size_t i = 0; i < actual.size(); ++i) {
                ASSERT_TRUE(std::isfinite(actual[i]));
                EXPECT_NEAR(actual[i], reference[i],
                    (std::is_same_v<T, float> ? 2e-5 : 2e-13) * (1 + std::abs(reference[i])));
            }
            if (threads == 1) {
                one = actual;
            } else {
                EXPECT_EQ(one, actual);
            }
        }
    }
}
TEST(SymmetricLocality, DoubleContracts) {
    check_symmetric_locality<double, int64_t>();
    check_symmetric_locality<double, int32_t>();
}

TEST(SymmetricLocality, FloatContracts) {
    check_symmetric_locality<float, int64_t>();
    check_symmetric_locality<float, int32_t>();
}

TEST(SymmetricLocality, EmptyWindowDoesNotReadMatrix) {
    const int64_t n = 129, d = 7;
    RandBLAS::sparse_data::COOMatrix<double> S(d + 2, n + 2);
    S.reserve(n);
    for (int64_t p = 0; p < n; ++p) {
        S.rows[p] = 0;
        S.cols[p] = p;
        S.vals[p] = 1.;
    }
    std::vector<double> B(d * n, 3.);
    // All entries lie before the requested row window. A must not be read.
    RandBLAS::sparse_data::coo_lsksys(blas::Layout::RowMajor,
        blas::Uplo::Upper, d, n, 2., S, 1, 0,
        static_cast<const double *>(nullptr), n, B.data(), n);
    for (double x : B) EXPECT_EQ(x, 3.);
}

} // namespace
