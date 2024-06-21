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

#ifndef randblas_util_hh
#define randblas_util_hh

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
void genmat(
	int64_t n_rows,
	int64_t n_cols,
	T* mat,
	uint64_t seed)
{
	typedef r123::Philox2x64 CBRNG;
	CBRNG::key_type key = {{seed}};
	CBRNG::ctr_type ctr = {{0,0}};
	CBRNG g;
	uint64_t prod = n_rows * n_cols;
	for (uint64_t i = 0; i < prod; ++i)
	{
		ctr[0] = i;
		CBRNG::ctr_type rand = g(ctr, key);
		mat[i] = r123::uneg11<T>(rand.v[0]);
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

template<typename RNG>
bool compare_ctr(typename RNG::ctr_type c1, typename RNG::ctr_type c2) {
    int len = c1.size();
    
    for (int ind = len - 1; ind >= 0; ind--) {
        if (c1[ind] > c2[ind]) {
            return true;
        } else if (c1[ind] < c2[ind]) {
            return false;
        }
    }
    return false;
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
void require_symmetric(blas::Layout layout, T* A, int64_t n, int64_t lda, T tol) { 
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
                randblas_error_if_msg(viol > rel_tol, message, i, j, j, i, viol, rel_tol);
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


} // end namespace RandBLAS::util

#endif
