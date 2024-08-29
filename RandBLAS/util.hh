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
#include <blas.hh>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include <iostream>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>


namespace RandBLAS::util {

template <typename T>
void safe_scal(int64_t n, T a, T* x, int64_t inc_x) {
    if (a == 0.0) {
        for (int64_t i = 0; i < n; ++i)
            x[i*inc_x] = 0.0;
    } else {
        blas::scal(n, a, x, inc_x);
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
/// Discussion.
/// @endverbatim
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

enum ArrayStyle : char {
    MATLAB = 'M',
    Python = 'P'
};

// =============================================================================
/// \fn print_colmaj(int64_t n_rows, int64_t n_cols, T *a, cout_able &label, 
///     ArrayStyle style = ArrayStyle::MATLAB, 
///     std::ios_base::fmtflags &flags = std::cout.flags()
/// )
/// @verbatim embed:rst:leading-slashes
/// Notes: see https://cplusplus.com/reference/ios/ios_base/fmtflags/ for info on the optional flags argument. d
/// @endverbatim
template <typename T, typename cout_able>
void print_colmaj(
    int64_t n_rows, int64_t n_cols, T *a, cout_able &label,
    ArrayStyle style = ArrayStyle::MATLAB, 
    const std::ios_base::fmtflags flags = std::cout.flags()
) {
    std::string abs_start {(style == ArrayStyle::MATLAB) ? "\n\t[ "  : "\nnp.array([\n\t[ " };
    std::string mid_start {(style == ArrayStyle::MATLAB) ? "\t  "    : "\t[ "               };
    std::string mid_end   {(style == ArrayStyle::MATLAB) ? "; ...\n" : "],\n"               };
    std::string abs_end   {(style == ArrayStyle::MATLAB) ? "];\n"    : "]\n])\n"            };

	int64_t i, j;
    T val;
    auto old_flags = std::cout.flags();
    std::cout.flags(flags);
	std::cout << std::endl << label << abs_start << std::endl;
    for (i = 0; i < n_rows; ++i) {
        std::cout << mid_start;
        for (j = 0; j < n_cols - 1; ++j) {
            val = a[i + n_rows * j];
            std::cout << "  " << val << ","; 
        }
        // j = n_cols - 1
        val = a[i + n_rows * j];
        std::cout << "  " << val;
        if (i < n_rows - 1) {
           std::cout << mid_end;
        } else {
            std::cout << abs_end;
        }
    }
    std::cout.flags(old_flags);
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
/// Discussion.
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
///     int64_t n, int64_t strict_offset, T val,  T* A, int64_t lda
/// )
/// @verbatim embed:rst:leading-slashes
/// Discussion.
/// @endverbatim
template <typename T>
void overwrite_triangle(blas::Layout layout, blas::Uplo to_overwrite, int64_t n, int64_t strict_offset, T val,  T* A, int64_t lda) {
    auto [inter_row_stride, inter_col_stride] = layout_to_strides(layout, lda);
    #define matA(_i, _j) A[(_i)*inter_row_stride + (_j)*inter_col_stride]
    if (to_overwrite == blas::Uplo::Upper) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + strict_offset; j < n; ++j) {
                matA(i,j) = val;
            }
        }
    } else if (to_overwrite == blas::Uplo::Lower) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + strict_offset; j < n; ++j) {
                matA(j,i) = val;
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
/// In-place transpose of square matrix of order n, with leading dimension lda.
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
/// \fn weights_to_cdf(int64_t n, T* w, T error_if_below = -std::numeric_limits<T>::epsilon())
/// @verbatim embed:rst:leading-slashes
/// Discussion.
/// @endverbatim
template <typename T>
void weights_to_cdf(int64_t n, T* w, T error_if_below = -std::numeric_limits<T>::epsilon()) {
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
static inline TO uneg11_to_uneg01(TI in) {
    return ((TO) in + (TO) 1.0)/ ((TO) 2.0);
}

// =============================================================================
/// \fn sample_indices_iid(int64_t n, TF* cdf, int64_t k, int64_t* samples, RNGState<RNG> state)
/// @verbatim embed:rst:leading-slashes
/// cdf represents a cumulative distribution function over {0, ..., n - 1}.
/// TF is a template parameter for a real floating point type.
/// We overwrite the "samples" buffer with k (independent) samples from the
/// distribution specified by cdf.
/// @endverbatim
template <typename TF, typename RNG>
RNGState<RNG> sample_indices_iid(int64_t n, TF* cdf, int64_t k, int64_t* samples, RNGState<RNG> state) {
    auto [ctr, key] = state;
    RNG gen;
    auto rv_array = r123ext::uneg11::generate(gen, ctr, key);
    int64_t len_c = (int64_t) state.len_c;
    int64_t rv_index = 0;
    for (int64_t i = 0; i < k; ++i) {
        if ((i+1) % len_c == 1) {
            ctr.incr(1);
            rv_array = r123ext::uneg11::generate(gen, ctr, key);
            rv_index = 0;
        }
        auto random_unif01 = uneg11_to_uneg01<TF>(rv_array[rv_index]);
        int64_t sample_index = std::lower_bound(cdf, cdf + n, random_unif01) - cdf;
        samples[i] = sample_index;
        rv_index += 1;
    }
    return RNGState<RNG>(ctr, key);
}
 
template <typename RNG, typename T, bool WriteRademachers = true, SignedInteger sint_t = int64_t>
RNGState<RNG> sample_indices_iid_uniform(int64_t n,  int64_t k, sint_t* samples, T* rademachers, RNGState<RNG> state) {
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
        auto random_unif01 = uneg11_to_uneg01<double>(rv_array[rv_index]);
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
    return RNGState<RNG>(ctr, key);
}


// =============================================================================
/// \fn sample_indices_iid_uniform(int64_t n,  int64_t k, sint_t* samples, RNGState<RNG> state)
/// @verbatim embed:rst:leading-slashes
/// Overwrite the "samples" buffer with k (independent) samples from the
/// uniform distribution over {0, ..., n - 1}.
/// @endverbatim
template <typename RNG, SignedInteger sint_t = int64_t>
RNGState<RNG> sample_indices_iid_uniform(int64_t n,  int64_t k, sint_t* samples, RNGState<RNG> state) {
    return sample_indices_iid_uniform<RNG,float,false,sint_t>(n, k, samples, (float*) nullptr, state);
}


} // end namespace RandBLAS

