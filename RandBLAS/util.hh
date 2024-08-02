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
#include <cstdio>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>

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
void print_colmaj(int64_t n_rows, int64_t n_cols, T *a, char label[])
{
	int64_t i, j;
    T val;
	std::cout << "\n" << label << std::endl;
    for (i = 0; i < n_rows; ++i) {
        std::cout << "\t";
        for (j = 0; j < n_cols - 1; ++j) {
            val = a[i + n_rows * j];
            if (val < 0) {
				//std::cout << string_format("  %2.4f,", val);
                printf("  %2.20f,", val);
            } else {
				//std::cout << string_format("   %2.4f", val);
				printf("   %2.20f,", val);
            }
        }
        // j = n_cols - 1
        val = a[i + n_rows * j];
        if (val < 0) {
   			//std::cout << string_format("  %2.4f,", val); 
			printf("  %2.20f,", val);
		} else {
            //std::cout << string_format("   %2.4f,", val);
			printf("   %2.20f,", val);
		}
        printf("\n");
    }
    printf("\n");
    return;
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

template <typename T>
void symmetrize(blas::Layout layout, blas::Uplo uplo, T* A, int64_t n, int64_t lda) { 

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
                std::string message = "Symmetry check failed. |A(%i,%i) - A(%i,%i)| was %d, which exceeds tolerance of %d.";
                auto _message = message.c_str();
                randblas_error_if_msg(viol > rel_tol, _message, i, j, j, i, viol, rel_tol);
            }
        }
    }
    #undef matA
    return;
}

/**
 * In-place transpose of square matrix of order n, with leading dimension lda.
 * Turns out that "layout" doesn't matter here.
*/
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

/***
 * cdf represents a cumulative distribution function over {0, ..., n - 1}.
 * 
 * TF is a template parameter for a real floating point type.
 * 
 * We overwrite the "samples" buffer with k (independent) samples from the
 * distribution specified by cdf.
 */
template <typename TF, typename int64_t, typename RNG>
RNGState<RNG> sample_indices_iid(
    int64_t n, TF* cdf, int64_t k, int64_t* samples, RNGState<RNG> state
) {
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

/*** 
 * Overwrite the "samples" buffer with k (independent) samples from the
 * uniform distribution over {0, ..., n - 1}.
 */
template <typename int64_t, typename RNG>
RNGState<RNG> sample_indices_iid_uniform(
    int64_t n,  int64_t k, int64_t* samples, RNGState<RNG> state
) {
    auto [ctr, key] = state;
    RNG gen;
    auto rv_array = r123ext::uneg11::generate(gen, ctr, key);
    int64_t len_c = (int64_t) state.len_c;
    int64_t rv_index = 0;
    double dN = (double) n;
    for (int64_t i = 0; i < k; ++i) {
        if ((i+1) % len_c == 1) {
            ctr.incr(1);
            rv_array = r123ext::uneg11::generate(gen, ctr, key);
            rv_index = 0;
        }
        auto random_unif01 = uneg11_to_uneg01<double>(rv_array[rv_index]);
        int64_t sample_index = (int64_t) dN * random_unif01;
        samples[i] = sample_index;
        rv_index += 1;
    }
    return RNGState<RNG>(ctr, key);
}


} // end namespace RandBLAS::util
