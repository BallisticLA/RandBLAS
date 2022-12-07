#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_TESTING_UTIL_HH
#define RandBLAS_TESTING_UTIL_HH

namespace RandBLAS_Testing::Util {

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
);

template <typename T>
void buffs_approx_equal(
    const T *actual_ptr,
    const T *expect_ptr,
    int64_t size
);


}  // end namespace RandBLAS_Testing::Util

#endif
