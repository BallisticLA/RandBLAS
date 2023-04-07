#ifndef randblas_test_util_hh
#define randblas_test_util_hh

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <iostream>

namespace RandBLAS_Testing::Util {

/** Tests two floating point numbers for approximate equality.
 * See https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *
 * @param[in] A    one number to compare
 * @param[in] B    the second number to compare
 * @param[in] atol is an absolute tolerance that comes into play when
 *                 the values are close to zero
 * @param[in] rtol is a relative tolerance, which should be close to
 *                 epsilon for the given type.
 * @param[inout] str a stream to send a decritpive error message to
 *
 * @returns true if the numbers are atol absolute difference or rtol relative
 *          difference from each other.
 */
template <typename T>
bool approx_equal(T A, T B, std::ostream &str,
    T atol = T(10)*std::numeric_limits<T>::epsilon(),
    T rtol = std::numeric_limits<T>::epsilon())
{
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    T diff_ab = abs(A - B);
    if (diff_ab <= atol)
        return true;

    T max_ab = std::max(abs(B), abs(A));

    if (diff_ab <= max_ab * rtol)
        return true;

    str.precision(std::numeric_limits<T>::max_digits10);

    str << A << " != " << B << " with absDiff=" << diff_ab
        << ", relDiff=" << max_ab*rtol << ", atol=" << atol
        << ", rtol=" << rtol;

    return false;
}



/** Test two arrays are approximately equal elementwise.
 *
 * @param[in] actual_ptr the array to check
 * @param[in] expect_ptr the array to check against
 * @param[in] size the number of elements to compare
 * @param[in] testName the name of the test, used in decriptive message
 * @param[in] fileName the name of the file, used in descriptive message
 * @param[in] lineNo the line tested, used in descriptive message
 * 
 * aborts if any of the elemnts are not approximately equal.
 */
template <typename T>
void buffs_approx_equal(
    const T *actual_ptr,
    const T *expect_ptr,
    int64_t size,
    const char *testName,
    const char *fileName,
    int lineNo,
    T atol = T(10)*std::numeric_limits<T>::epsilon(),
    T rtol = std::numeric_limits<T>::epsilon()
) {
    std::ostringstream oss;
    for (int64_t i = 0; i < size; ++i) {
        if (!approx_equal(actual_ptr[i], expect_ptr[i], oss, atol, rtol)) {
            FAIL() << std::endl << fileName << ":" << lineNo << std::endl
                << testName << std::endl << "Test failed at index " << i
                << " " << oss.str() << std::endl;
            oss.str("");
        }
    }
}

template <typename T>
void buffs_approx_equal(
    const T *actual_ptr,
    const T *expect_ptr,
    const T *bounds_ptr,
    int64_t size,
    const char *test_name,
    const char *file_name,
    int line_no
) {
    std::ostringstream oss;
    for (int64_t i = 0; i < size; ++i) {
        T actual_err = abs(actual_ptr[i] - expect_ptr[i]);
        T allowed_err = bounds_ptr[i];
        if (actual_err > allowed_err) {
            FAIL() << std::endl << file_name << ":" << line_no << std::endl
                    << test_name << std::endl << "Test failed at index "
                    << i << oss.str() << std::endl;
            oss.str("");
        }
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
    int64_t ldb,
    const char *testName,
    const char *fileName,
    int lineNo,
    T atol = T(10)*std::numeric_limits<T>::epsilon(),
    T rtol = std::numeric_limits<T>::epsilon()
) {
    std::ostringstream oss;
    // check that A == op(B), where A is m-by-n.
    auto idxa = [lda, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*lda) : (j + i*lda);
    };
    auto idxb = [ldb, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*ldb) : (j + i*ldb);
    };
    if (transB == blas::Op::NoTrans) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (!approx_equal(A[idxa(i, j)], B[idxb(i, j)], oss, atol, rtol)) {
                    FAIL() << std::endl << fileName << ":" << lineNo << std::endl
                        << testName << std::endl << "Test failed at index ("
                        << i << ", " << j << ") " << oss.str() << std::endl;
                    oss.str("");
                }
            }
        }
    } else {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (!approx_equal(A[idxa(i, j)], B[idxb(j, i)], oss, atol, rtol)) {
                    FAIL() << std::endl << fileName << ":" << lineNo << std::endl
                        << testName << std::endl << "Test failed at index ("
                        << j << ", " << i << ") "  << oss.str() << std::endl;
                    oss.str("");
                }
            }
        }
    }
}

} // end namespace RandBLAS_Testing::Util

#endif
