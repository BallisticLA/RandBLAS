#ifndef randblas_test_util_hh
#define randblas_test_util_hh

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <iostream>

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

#include <RandBLAS/dense.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/util.hh>
#include <math.h>


namespace RandBLAS_Testing::Util {


//Function that fills in a random matrix and truncates at the end of each row so that each row starts with a fresh counter.
template<typename T, typename RNG, typename OP>
static void fill_dense_rmat_trunc(
    T* mat,
    int64_t n_rows,
    int64_t n_cols,
    const RandBLAS::RNGState<RNG> & seed
) {

    RNG rng;
    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type k = seed.key;
    
    int ind = 0;
    int cts = n_cols / RNG::ctr_type::static_size; //number of counters per row, where all the random numbers are to be filled in the array.
    int res = n_cols % RNG::ctr_type::static_size; //Number of random numbers to be filled at the end of each row the the last counter of the row

    for (int i = 0; i < n_rows; i++) {
        for (int ctr = 0; ctr < cts; ctr++){
            auto rv = OP::generate(rng, c, k);
            for (int j = 0; j < RNG::ctr_type::static_size; j++) {
                mat[ind] = rv[j];
                ind++;
            }
            c.incr();
        }
        if (res != 0) { 
            for (int j = 0; j < res; j++) {
                auto rv = OP::generate(rng, c, k);
                mat[ind] = rv[j];
                ind++;
            }
            c.incr();
        }
    }
}


template <class T>
std::string type_name() { // call as type_name<obj>()
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}


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
            FAIL() << std::endl << "\t" <<  file_name << ":" << line_no << std::endl
                    << "\t" << test_name << std::endl << "\tTest failed at index "
                    << i << ".\n\t| (" << actual_ptr[i] << ") - (" << expect_ptr[i] << ") | "
                    << " > " << allowed_err << oss.str() << std::endl;
            oss.str("");
        }
    }
}

template <typename T>
void matrices_approx_equal(
    blas::Layout layoutA,
    blas::Layout layoutB,
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
    auto idxa = [lda, layoutA](int64_t i, int64_t j) {
        return  (layoutA == blas::Layout::ColMajor) ? (i + j*lda) : (j + i*lda);
    };
    auto idxb = [ldb, layoutB](int64_t i, int64_t j) {
        return  (layoutB == blas::Layout::ColMajor) ? (i + j*ldb) : (j + i*ldb);
    };
    if (transB == blas::Op::NoTrans) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (!approx_equal(A[idxa(i, j)], B[idxb(i, j)], oss, atol, rtol)) {
                    FAIL() << std::endl << fileName << ":" << lineNo << std::endl
                        << testName << std::endl << "\tTest failed at index ("
                        << i << ", " << j << ")\n\t" << oss.str() << std::endl;
                    oss.str("");
                }
            }
        }
    } else {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (!approx_equal(A[idxa(i, j)], B[idxb(j, i)], oss, atol, rtol)) {
                    FAIL() << std::endl << fileName << ":" << lineNo << std::endl
                        << testName << std::endl << "\tTest failed at index ("
                        << j << ", " << i << ")\n\t"  << oss.str() << std::endl;
                    oss.str("");
                }
            }
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
    matrices_approx_equal(layout, layout, transB, m, n, A, lda, B, ldb, testName, fileName, lineNo, atol, rtol);
}

template <typename T, typename RNG, RandBLAS::SignedInteger sint_t>
void sparseskop_to_dense(
    RandBLAS::SparseSkOp<T, RNG, sint_t> &S0,
    T *mat,
    blas::Layout layout
) {
    RandBLAS::SparseDist D = S0.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;
    auto idx = [D, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*D.n_rows) : (j + i*D.n_cols);
    };
    int64_t nnz;
    if (D.major_axis == RandBLAS::MajorAxis::Short) {
        nnz = D.vec_nnz * MAX(D.n_rows, D.n_cols);
    } else {
        nnz = D.vec_nnz * MIN(D.n_rows, D.n_cols);
    }
    for (int64_t i = 0; i < nnz; ++i) {
        sint_t row = S0.rows[i];
        sint_t col = S0.cols[i];
        T val = S0.vals[i];
        mat[idx(row, col)] = val;
    }
}

} // end namespace RandBLAS_Testing::Util
#endif
