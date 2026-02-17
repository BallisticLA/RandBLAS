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

#pragma once

#include <RandBLAS/base.hh>
#include <RandBLAS/exceptions.hh>
#include <RandBLAS/compilers.hh>
#include <blas.hh>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include <iostream>
#include <iomanip>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <sstream>
#include <numeric>
#include <cstdlib>


namespace RandBLAS::util {

template <typename T>
void safe_scal(int64_t n, T a, T* x) {
    if (a == 0.0) {
        std::fill(x, x + n, static_cast<T>(0));
    } else {
        blas::scal(n, a, x, static_cast<int64_t>(1));
    }
}

template <typename T>
void omatcopy(int64_t m, int64_t n, const T* A, int64_t irs_a, int64_t ics_a, T* B, int64_t irs_b, int64_t ics_b) {
    // TODO:
    //     1. Order the loops with consideration to cache efficiency.
    //     2. Vectorize one of the loops with blas::copy or std::memcpy.
    #define MAT_A(_i, _j) A[(_i)*irs_a + (_j)*ics_a]
    #define MAT_B(_i, _j) B[(_i)*irs_b + (_j)*ics_b]
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            MAT_B(i,j) = MAT_A(i,j);
        }
    }
    #undef MAT_A
    #undef MAT_B
    return;
}

template <typename T>
void flip_layout(blas::Layout layout_in, int64_t m, int64_t n, std::vector<T> &A, int64_t lda_in, int64_t lda_out) {
    using blas::Layout;
    Layout layout_out;
    int64_t len_buff_A_out;
    if (layout_in == Layout::ColMajor) {
        layout_out = Layout::RowMajor;
        randblas_require(lda_in  >= m);
        randblas_require(lda_out >= n);
        len_buff_A_out = lda_out * m;
    } else {
        layout_out = Layout::ColMajor;
        randblas_require(lda_in  >= n);
        randblas_require(lda_out >= m);
        len_buff_A_out = lda_out * n;
    }
    // irs = inter row stride (stepping down a column)
    // ics = inter column stride (stepping across a row)
    auto [irs_in,   ics_in] = layout_to_strides(layout_in,  lda_in);
    auto [irs_out, ics_out] = layout_to_strides(layout_out, lda_out);

    if (len_buff_A_out >= (int64_t) A.size()) {
        A.resize(len_buff_A_out);
    }
    std::vector<T> A_in(A);
    T* A_buff_in  = A_in.data();
    T* A_buff_out = A.data();
    omatcopy(m, n, A_buff_in, irs_in, ics_in, A_buff_out, irs_out, ics_out);
    A.erase(A.begin() + len_buff_A_out, A.end());
    A.resize(len_buff_A_out);
    return;
}

// =============================================================================
/// \fn require_symmetric(blas::Layout layout, const T* A, int64_t n, int64_t lda, T tol)
/// @verbatim embed:rst:leading-slashes
/// If :math:`\ttt{tol} \geq 0`, this function checks if
///
///       .. math::
///           \frac{|A[i + j \cdot \lda] - A[i \cdot \lda + j]|}{|A[i + j \cdot \lda]| + |A[i \cdot \lda + j]| + 1} \leq \ttt{tol}
///
/// for all :math:`i,j \in \\{0,\ldots,n-1\\}.` An error is raised if any such check fails.
/// This function returns immediately without performing any checks if :math:`\ttt{tol} < 0.`
/// @endverbatim
/// sketch_symmetric calls this function with \math{\ttt{tol} = 0} by default.
/// 
template <typename T>
void require_symmetric(blas::Layout layout, const T* A, int64_t n, int64_t lda, T tol) { 
    if (tol < 0)
        return;
    auto [inter_row_stride, inter_col_stride] = layout_to_strides(layout, lda);
    #define matA(_i, _j) A[(_i)*inter_row_stride + (_j)*inter_col_stride]
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i+1; j < n; ++j) {
            T Aij = matA(i,j);
            T Aji = matA(j,i);
            T viol = abs(Aij - Aji);
            T rel_tol = (abs(Aij) +  abs(Aji) + 1)*tol;
            if (viol > rel_tol) {
                std::string message = "Symmetry check failed. |A(%i,%i) - A(%i,%i)| was %e, which exceeds tolerance of %e.";
                auto _message = message.c_str();
                randblas_error_if_msg(viol > rel_tol, _message, i, j, j, i, viol, rel_tol);
            }
        }
    }
    #undef matA
    return;
}

} // end namespace RandBLAS::util

namespace RandBLAS {


// =============================================================================
/// Specifies whether string representations of matrices should use
/// MATLAB-style or Python-style formatting. You should be able to copy
/// the output of an array printed in a given style and paste it directly
/// into the corresponding programming language's interpreter.
/// 
enum ArrayStyle : char {
    MATLAB = 'M',
    Python = 'P'
};


namespace detail {
RandBLAS_OPTIMIZE_OFF
// ^ It would be extra bad if a compiler somehow messed up the functions in
//   this `detail` namespace.

// -----------------------------------------------------------------------------
// RandBLAS 1.0 introduced a print_buff_to_stream function where `label`
// was templated as `typename cout_able`. This meant that passing
// string literals to `label` resulted in separate template instantiations 
// of `const char[<string length here>]` for different string lengths.
//
// Using this print_buff_to_stream_impl function lets us keep the original
// API while ensuring the compiler generates code for std::string.
//
template <typename T>
void print_buff_to_stream_impl(
    std::ostream &stream, int64_t n_rows, int64_t n_cols, T *A,
    int64_t irs, int64_t ics, const std::string &label, int decimals, ArrayStyle style
) {
    std::string abs_start, mid_start, mid_end, abs_end;
    if (style == ArrayStyle::MATLAB) {
        abs_start = " = [ ... \n";
        mid_start = "\t"; mid_end = " ; ...\n";
        abs_end   = " ; ...\n];\n";
    } else {
        abs_start = " = np.array([\n";
        mid_start = "\t["; mid_end = " ],\n";
        abs_end =  " ]\n])\n";
    }

	int64_t i, j;
    T val;
	stream << std::endl << label << abs_start;
    stream << std::setprecision(decimals);
    for (i = 0; i < n_rows; ++i) {
        stream << mid_start;
        for (j = 0; j < n_cols - 1; ++j) {
            val = A[i*irs + j*ics];
            stream << "  " << val << ",";
        }
        // j = n_cols - 1
        val = A[i*irs + j*ics];
        stream << "  " << val;
        if (i < n_rows - 1) {
           stream << mid_end;
        } else {
            stream << abs_end;
        }
    }
    stream << std::endl;
    return;
}

// -----------------------------------------------------------------------------
// We need this overload of print_buff_to_stream_impl to hold the branching that
// occured in the print_buff_to_stream overload with the `layout` parameter.
template <typename T>
void print_buff_to_stream_impl(
    std::ostream &stream, blas::Layout layout, int64_t n_rows, int64_t n_cols, T *A,
    int64_t lda, const std::string &label, int decimals, ArrayStyle style
) {
    int64_t irs, ics;
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= n_rows);
        irs = 1; ics = lda;
    } else {
        randblas_require(lda >= n_cols);
        irs = lda; ics = 1;
    }
    print_buff_to_stream_impl<T>(stream, n_rows, n_cols, A, irs, ics, label, decimals, style);
    return;
}

RandBLAS_OPTIMIZE_ON
} // namespace detail

// =============================================================================
/// \fn print_buff_to_stream(
///     std::ostream &stream, int64_t n_rows, int64_t n_cols, T *A,
///     int64_t irs, int64_t ics, cout_able &label, int decimals, ArrayStyle style
/// )
/// Writes a string representation of \math{\mat(A)} to the provided stream.
/// The first line of the output will be \math{\ttt{label} + " = "} followed by a style-specific
/// representation of the matrix (MATLAB style or NumPy/Python style).
///
/// The matrix \math{\mat(A)} is defined by reading from \math{\ttt{A}} with
/// inter-row stride \math{\ttt{irs}} and inter-column stride \math{\ttt{ics}.}
/// That means \math{\mat(A)_{ij} = \ttt{A}[i*\ttt{irs} + j*\ttt{ics}],} using
/// zero-indexing.
///
/// Note: \math{\ttt{std::cout}} is a good choice of stream in most situations.
/// We don't set a default value since it's better to be explicit in this context.
///
template <typename T, typename cout_able = std::string>
void print_buff_to_stream(
    std::ostream &stream, int64_t n_rows, int64_t n_cols, T *A,
    int64_t irs, int64_t ics, cout_able &label, int decimals = 8, ArrayStyle style = ArrayStyle::MATLAB
) {
    std::ostringstream ss;
    ss << label;
    detail::print_buff_to_stream_impl<T>(stream, n_rows, n_cols, A, irs, ics, ss.str(), decimals, style);
}

// =============================================================================
/// \fn print_buff_to_stream(
///     std::ostream &stream, blas::Layout layout, int64_t n_rows, int64_t n_cols, T *A,
///     int64_t lda, cout_able &label, int decimals, ArrayStyle style
/// )
/// Writes a string representation of \math{\mat(A)} to the provided stream.
/// The first line of the output will be \math{\ttt{label},} followed by a style-specific
/// representation of the matrix (MATLAB style or NumPy/Python style).
///
template <typename T, typename cout_able = std::string>
void print_buff_to_stream(
    std::ostream &stream, blas::Layout layout, int64_t n_rows, int64_t n_cols, T *A,
    int64_t lda, cout_able &label, int decimals = 8, ArrayStyle style = ArrayStyle::MATLAB
) {
    std::ostringstream ss;
    ss << label;
    detail::print_buff_to_stream_impl<T>(stream, layout, n_rows, n_cols, A, lda, ss.str(), decimals, style);
}

// This function is here for compatibility with existing RandLAPACK code.
// New code should use print_buff_to_stream.
template <typename T, typename cout_able = std::string>
inline void print_colmaj(
    int64_t n_rows, int64_t n_cols, T *A, cout_able &label, int decimals = 8,
    ArrayStyle style = ArrayStyle::MATLAB
) {
    print_buff_to_stream(std::cout, blas::Layout::ColMajor, n_rows, n_cols, A, label, decimals, style);
    return;
}

// =============================================================================
/// \fn typeinfo_as_string()
/// @verbatim embed:rst:leading-slashes
/// When called as ``typeinfo_as_string<your_variable>()``, this function returns a string 
/// giving all available type information for ``your_variable``. This can be useful
/// for inspecting types in the heretical practice of *print statement debugging*.
/// @endverbatim
template <class T>
std::string typeinfo_as_string() {
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

// =============================================================================
/// \fn symmetrize(blas::Layout layout, blas::Uplo uplo, int64_t n, T* A, int64_t lda)
/// @verbatim embed:rst:leading-slashes
/// Use this function to convert a matrix that BLAS can *interpet* as symmetric into a matrix
/// that's explicitly symmetric.
///
/// Formally, :math:`A` points to the start of a buffer for an :math:`n \times n` matrix :math:`\mat(A)`
/// stored in :math:`\ttt{layout}` order with leading dimension :math:`\ttt{lda}.`
/// This function copies the strict part of the :math:`\ttt{uplo}` triangle of :math:`\mat(A)`
/// into the strict part of the opposing triangle.
///
/// @endverbatim
template <typename T>
void symmetrize(blas::Layout layout, blas::Uplo uplo, int64_t n, T* A, int64_t lda) { 
    auto [inter_row_stride, inter_col_stride] = layout_to_strides(layout, lda);
    #define matA(_i, _j) A[(_i)*inter_row_stride + (_j)*inter_col_stride]
    if (uplo == blas::Uplo::Upper) {
        // copy to lower
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i+1; j < n; ++j) {
                matA(j,i) = matA(i,j);
            }
        }
    } else if (uplo == blas::Uplo::Lower) {
        // copy to upper
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i+1; j < n; ++j) {
                matA(i,j) = matA(j,i);
            }
        }
    }
    #undef matA
    return;
}

// =============================================================================
/// \fn overwrite_triangle(blas::Layout layout, blas::Uplo to_overwrite,
///     int64_t n, int64_t k,  T* A, int64_t lda
/// )
/// @verbatim embed:rst:leading-slashes
/// Use this function to convert a matrix which BLAS can *interpret* as triangular into a matrix that's
/// explicitly triangular.
/// 
/// Formally, :math:`A` points to the start of a buffer for an :math:`n \times n` matrix :math:`\mat(A)`
/// stored in :math:`\ttt{layout}` order with leading dimension :math:`\ttt{lda},`
/// and :math:`\ttt{k}` is a nonnegative integer.
///
/// This function overwrites :math:`A` so that ...
///  * If :math:`\ttt{to_overwrite} = \ttt{Uplo::Lower},` then elements of :math:`\mat(A)` on or
///    below its :math:`\ttt{k}^{\text{th}}` subdiagonal are overwritten with zero.
///  * If :math:`\ttt{to_overwrite} = \ttt{Uplo::Upper},` then elements of :math:`\mat(A)` on or
///    above its :math:`\ttt{k}^{\text{th}}` superdiagonal are overwritten with zero.
///
/// This function raises an error if :math:`\ttt{k}` is negative or if 
/// :math:`\ttt{to_overwrite}` is neither Upper nor Lower.
///
/// @endverbatim
template <typename T>
void overwrite_triangle(blas::Layout layout, blas::Uplo to_overwrite, int64_t n, int64_t k,  T* A, int64_t lda) {
    auto [inter_row_stride, inter_col_stride] = layout_to_strides(layout, lda);
    #define matA(_i, _j) A[(_i)*inter_row_stride + (_j)*inter_col_stride]
    if (to_overwrite == blas::Uplo::Upper) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + k; j < n; ++j) {
                matA(i,j) = 0.0;
            }
        }
    } else if (to_overwrite == blas::Uplo::Lower) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + k; j < n; ++j) {
                matA(j,i) = 0.0;
            }
        }
    } else {
        throw std::runtime_error("Invalid argument for UPLO.");
    }
    #undef matA
    return;
}

// =============================================================================
/// \fn transpose_square(T* A, int64_t n, int64_t lda)
/// @verbatim embed:rst:leading-slashes
/// In-place transpose of square matrix of order :math:`n`, with leading dimension :math:`\ttt{lda}.`
///
/// It turns out that there's no implementation difference between row-major
/// or column-major data, so we don't accept a layout parameter.
/// @endverbatim
template <typename T>
void transpose_square(T* A, int64_t n, int64_t lda) {
    #define matA(_i, _j) A[(_i) + lda*(_j)]
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i+1; j < n; ++j) {
            std::swap(matA(i,j), matA(j,i));
        }
    }
    #undef matA
    return;
}

// =============================================================================
/// \fn sqrt_epsilon()
/// Alias for sqrt(numeric_limits<T>::epsilon()). For example,
/// \math{\ttt{sqrt_epsilon<float>()} \approx 0.0003452}, and 
/// \math{\ttt{sqrt_epsilon<double>()} \approx 1.4901\text{e-}8}.
/// 
template <typename T>
T sqrt_epsilon() {
    return std::sqrt(std::numeric_limits<T>::epsilon());
}

// =============================================================================
/// \fn weights_to_cdf(int64_t n, T* w, T error_if_below = -sqrt_epsilon<T>())
/// @verbatim embed:rst:leading-slashes
/// Checks if all elements of length-:math:`n` array ":math:`w`" are at no smaller than 
/// :math:`\ttt{error_if_below}.` If this check passes, then we (implicitly) initialize :math:`v := w`` 
/// and overwrite :math:`w` by
///
/// .. math::
///
///     w_i = \frac{\textstyle\sum_{\ell=1}^{i}\max\{0, v_{\ell}\}}{\textstyle\sum_{j=1}^n \max\{0, v_j\}}.
/// 
/// @endverbatim
/// On exit, \math{w} is a CDF suitable for use with sample_indices_iid.
///
template <typename T>
void weights_to_cdf(int64_t n, T* w, T error_if_below = -sqrt_epsilon<T>()) {
    T sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        T val = w[i];
        randblas_require(val >= error_if_below);
        val = std::max(val, (T) 0.0);
        sum += val;
        w[i] = sum;
    }
    randblas_require(sum >= ((T) std::sqrt(n)) * std::numeric_limits<T>::epsilon());
    blas::scal(n, ((T)1.0) / sum, w, 1);
    return;
}

template <typename TO, typename TI>
static inline TO uneg11_to_u01(TI in) {
    return ((TO) in + (TO) 1.0)/ ((TO) 2.0);
}

// =============================================================================
/// @verbatim embed:rst:leading-slashes
/// :math:`\ttt{cdf}` encodes a cumulative distribution function over 
/// :math:`\{0, \ldots, n - 1\}.` For :math:`0 \leq i < n-1,` it satisfies
/// 
/// .. math::
///
///    0 \leq \ttt{cdf}[i] \leq \ttt{cdf}[i+1] \leq \ttt{cdf}[n-1] = 1.
/// 
/// On exit, :math:`\ttt{samples}` is overwritten by :math:`k` independent samples 
/// from :math:`\ttt{cdf}.` The returned RNGState should
/// be used for the next call to a random sampling function whose output should be statistically
/// independent from :math:`\ttt{samples}.`
/// @endverbatim
template <typename T, SignedInteger sint_t, typename state_t = RNGState<DefaultRNG>>
state_t sample_indices_iid(int64_t n, const T* cdf, int64_t k, sint_t* samples, const state_t &state) {
    auto [ctr, key] = state;
    using RNG = typename state_t::generator;
    RNG gen;
    auto rv_array = r123ext::uneg11::generate(gen, ctr, key);
    int64_t len_c = (int64_t) state.len_c;
    int64_t rv_index = 0;
    for (int64_t i = 0; i < k; ++i) {
        auto random_unif01 = uneg11_to_u01<T>(rv_array[rv_index]);
        sint_t sample_index = std::lower_bound(cdf, cdf + n, random_unif01) - cdf;
        samples[i] = sample_index;
        rv_index += 1;
        if (rv_index == len_c) {
            ctr.incr(1);
            rv_array = r123ext::uneg11::generate(gen, ctr, key);
            rv_index = 0;
        }
    }
    if (0 < rv_index) ctr.incr(1);
    return state_t(ctr, key);
}
 
template <typename T, SignedInteger sint_t, bool WriteRademachers = true, typename state_t = RNGState<DefaultRNG>>
state_t sample_indices_iid_uniform(int64_t n, int64_t k, sint_t* samples, T* rademachers, const state_t &state) {
    using RNG = typename state_t::generator;
    auto [ctr, key] = state;
    RNG gen;
    auto rv_array = r123ext::uneg11::generate(gen, ctr, key);
    int64_t len_c = static_cast<int64_t>(state.len_c);
    if constexpr (WriteRademachers) {
        len_c = 2*(len_c/2);
        // ^ round down to the nearest multiple of two.
    }
    int64_t rv_index = 0;
    double dN = (double) n;
    for (int64_t i = 0; i < k; ++i) {
        auto random_unif01 = uneg11_to_u01<double>(rv_array[rv_index]);
        sint_t sample_index = (sint_t) dN * random_unif01;
        samples[i] = sample_index;
        rv_index += 1;
        if constexpr (WriteRademachers) {
            rademachers[i] = (rv_array[rv_index] >= 0) ? (T) 1 : (T) -1;
            rv_index += 1;
        }
        if (rv_index == len_c) {
            ctr.incr(1);
            rv_array = r123ext::uneg11::generate(gen, ctr, key);
            rv_index = 0;
        }
    }
    if (0 < rv_index) ctr.incr(1);
    return state_t(ctr, key);
}


// =============================================================================
/// @verbatim embed:rst:leading-slashes
/// This function overwrites :math:`\ttt{samples}` with :math:`k` (independent) samples from the
/// uniform distribution over :math:`\{0, \ldots, n - 1\}.` The returned RNGState should
/// be used for the next call to a random sampling function whose output should be statistically
/// independent from :math:`\ttt{samples}.`
/// 
/// @endverbatim
template <SignedInteger sint_t = int64_t, typename state_t = RNGState<DefaultRNG>>
state_t sample_indices_iid_uniform(int64_t n,  int64_t k, sint_t* samples, const state_t &state) {
    return sample_indices_iid_uniform<float,sint_t,false,state_t>(n, k, samples, (float*) nullptr, state);
}


} // end namespace RandBLAS

