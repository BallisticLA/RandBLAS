#include <rbtutil.hh>

#include <gtest/gtest.h>
#include <math.h>
#include <numeric>

#define RELTOL_POWER 0.6
#define ABSTOL_POWER 0.75

namespace RandBLAS_Testing::Util {

template <typename T>
void buffs_approx_equal(
    const T *actual_ptr,
    const T *expect_ptr,
    int64_t size
) {
    T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
    for (int64_t i = 0; i < size; ++i) {
        T actual = actual_ptr[i];
        T expect = expect_ptr[i];
        T atol = reltol * std::min(abs(actual), abs(expect));
        EXPECT_NEAR(actual, expect, atol);
    }
}

template <typename T>
void matrices_approx_equal(
    blas::Layout layout,
    blas::Op transB,
    int64_t m,
    int64_t n,
    const T *A,
    int64_t lda,
    const T *B,
    int64_t ldb
) {
    // check that A == op(B), where A is m-by-n.
    T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
    auto idxa = [lda, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*lda) : (j + i*lda);
    };
    auto idxb = [ldb, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*ldb) : (j + i*ldb);
    };
    if (transB == blas::Op::NoTrans) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                T actual = A[idxa(i, j)];
                T expect = B[idxb(i, j)];
                T atol = reltol * std::min(abs(actual), abs(expect));
                EXPECT_NEAR(actual, expect, atol);
            }
        }
    } else {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                T actual = A[idxa(i, j)];
                T expect = B[idxb(j, i)];
                T atol = reltol * std::min(abs(actual), abs(expect));
                EXPECT_NEAR(actual, expect, atol);
            }
        }
    }
}

template void buffs_approx_equal<float>(const float *actual_ptr, const float *expect_ptr, int64_t size);
template void buffs_approx_equal<double>(const double *actual_ptr, const double *expect_ptr, int64_t size);

template void matrices_approx_equal<float>(blas::Layout layout, blas::Op transB, int64_t m, int64_t n, const float *A,
    int64_t lda, const float *B, int64_t ldb
);
template void matrices_approx_equal<double>(blas::Layout layout, blas::Op transB, int64_t m, int64_t n, const double *A,
    int64_t lda, const double *B, int64_t ldb
);

} // end namespace RandBLAS_Testing::Util
