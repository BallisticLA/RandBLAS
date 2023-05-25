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
#include <RandBLAS/sparse.hh>
#include <RandBLAS/ramm.hh>
#include <RandBLAS/util.hh>
#include <math.h>


namespace RandBLAS_Testing::Util {

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

template <typename T>
void sparseskop_to_dense(
    RandBLAS::sparse::SparseSkOp<T> &S0,
    T *mat,
    blas::Layout layout,
    bool take_abs = false
) {
    RandBLAS::sparse::SparseDist D = S0.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;
    auto idx = [D, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*D.n_rows) : (j + i*D.n_cols);
    };
    int64_t nnz;
    if (D.family == RandBLAS::sparse::SparsityPattern::SASO) {
        nnz = D.vec_nnz * MAX(D.n_rows, D.n_cols);
    } else {
        nnz = D.vec_nnz * MIN(D.n_rows, D.n_cols);
    }
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = S0.rows[i];
        int64_t col = S0.cols[i];
        T val = S0.vals[i];
        if (take_abs)
            val = abs(val);
        mat[idx(row, col)] = val;
    }
}

template <typename T, typename RNG>
void reference_lskges(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // mat(B) is d-by-n
    int64_t n, // op(mat(A)) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    RandBLAS::sparse::SparseSkOp<T,RNG> &S,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T beta,
    T *B,  // expected value produced by LSKGES; compute via GEMM.
    T *E,  // allowable floating point error; apply theory + compute by GEMM.
    int64_t ldb
){
    randblas_require(d > 0);
    randblas_require(m > 0);
    randblas_require(n > 0);
    std::vector<T> S_dense(S.dist.n_rows * S.dist.n_cols);
    sparseskop_to_dense<T>(S, S_dense.data(), layout, false);
    std::vector<T> S_dense_abs(S.dist.n_rows * S.dist.n_cols);
    sparseskop_to_dense<T>(S, S_dense_abs.data(), layout, true);

    // Dimensions of mat(A), rather than op(mat(A))
    int64_t rows_mat_A, cols_mat_A, rows_submat_S, cols_submat_S;
    if (transA == blas::Op::NoTrans) {
        rows_mat_A = m;
        cols_mat_A = n;
    } else {
        rows_mat_A = n;
        cols_mat_A = m;
    }
    // Dimensions of submat(S), rather than op(submat(S))
    if (transS == blas::Op::NoTrans) {
        rows_submat_S = d;
        cols_submat_S = m;
    } else {
        rows_submat_S = m;
        cols_submat_S = d;
    }
    // Sanity checks on dimensions and strides
    int64_t lds, pos, size_A, size_B;
    if (layout == blas::Layout::ColMajor) {
        lds = S.dist.n_rows;
        pos = i_os + lds * j_os;
        randblas_require(lds >= rows_submat_S);
        randblas_require(lda >= rows_mat_A);
        randblas_require(ldb >= d);
        size_A = lda * (cols_mat_A - 1) + rows_mat_A;;
        size_B = ldb * (n - 1) + d;
    } else {
        lds = S.dist.n_cols;
        pos = i_os * lds + j_os;
        randblas_require(lds >= cols_submat_S);
        randblas_require(lda >= cols_mat_A);
        randblas_require(ldb >= n);
        size_A = lda * (rows_mat_A - 1) + cols_mat_A;
        size_B = ldb * (d - 1) + n;
    }

    // Compute the reference value
    T* S_ptr = S_dense.data();
    blas::gemm(layout, transS, transA, d, n, m,
        alpha, &S_ptr[pos], lds, A, lda, beta, B, ldb
    );

    // Compute the matrix needed for componentwise error bounds.
    std::vector<T> A_abs_vec(size_A);
    T* A_abs = A_abs_vec.data();
    for (int64_t i = 0; i < size_A; ++i)
        A_abs[i] = abs(A[i]);
    if (beta != 0.0) {
        for (int64_t i = 0; i < size_B; ++i)
            E[i] = abs(B[i]);
    }
    T eps = std::numeric_limits<T>::epsilon();
    T err_alpha = (abs(alpha) * m) * (2 * eps);
    T err_beta = abs(beta) * eps;
    T* S_abs_ptr = S_dense_abs.data();
    blas::gemm(layout, transS, transA, d, n, m,
        err_alpha, &S_abs_ptr[pos], lds, A_abs, lda, err_beta, E, ldb
    );
    return;
}

template <typename T, typename RNG>
void reference_rskges(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transS,
    int64_t m, // B is m-by-d
    int64_t d, // op(S) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    RandBLAS::sparse::SparseSkOp<T,RNG> &S0,
    int64_t i_os,
    int64_t j_os,
    T beta,
    T *B, // expected value produced by LSKGES; compute via GEMM.
    T *E, // allowable floating point error; apply theory + compute by GEMM.
    int64_t ldb
) { 
    using blas::Layout;
    using blas::Op;
    //
    // Check dimensions of submat(S).
    //
    int64_t submat_S_rows, submat_S_cols;
    if (transS == Op::NoTrans) {
        submat_S_rows = n;
        submat_S_cols = d;
    } else {
        submat_S_rows = d;
        submat_S_cols = n;
    }
    randblas_require(submat_S_rows <= S0.dist.n_rows);
    randblas_require(submat_S_cols <= S0.dist.n_cols);
    //
    // Check dimensions of mat(A).
    //
    int64_t mat_A_rows, mat_A_cols;
    if (transA == Op::NoTrans) {
        mat_A_rows = m;
        mat_A_cols = n;
    } else {
        mat_A_rows = n;
        mat_A_cols = m;
    }
    if (layout == Layout::ColMajor) {
        randblas_require(lda >= mat_A_rows);
    } else {
        randblas_require(lda >= mat_A_cols);
    }
    //
    // Compute B = op(A) op(submat(S)) by LSKGES. We start with the identity
    //
    //      B^T = op(submat(S))^T op(A)^T
    //
    // Then we interchange the operator "op" for op(A) and the operator (*)^T.
    //
    //      B^T = op(submat(S))^T op(A^T)
    //
    // We tell LSKGES to process (B^T) and (A^T) in the opposite memory layout
    // compared to the layout for (A, B).
    // 
    auto trans_transS = (transS == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    reference_lskges(
        trans_layout, trans_transS, transA,
        d, m, n, alpha, S0, i_os, j_os, A, lda, beta, B, E, ldb
    );
}


template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v) {
    size_t n = v.size();
    os << "{";
    if (n)
    {
        os << v[0];
        for (size_t i = 1; i < n; ++i)
            os << ", " << v[i];
    }
    os << "}";
    return os;
}


} // end namespace RandBLAS_Testing::Util
#endif
