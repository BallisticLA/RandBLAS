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

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/skge.hh"

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>

#include <cmath>
#include <typeinfo>


namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;


// MARK: SUBMAT(S)

// =============================================================================
/// \fn sketch_vector(blas::Op opS, int64_t d, int64_t m, T alpha, SKOP &S,
///    int64_t ro_s, int64_t co_s, const T *x, int64_t incx, T beta, T *y, int64_t incy
/// )
/// @verbatim embed:rst:leading-slashes
/// Perform a GEMV-like operation. If :math:`{\opS} = \texttt{NoTrans},` then we perform
///
/// .. math::
///     \mat(y) = \alpha \cdot \underbrace{\submat(\mtxS)}_{d \times m} \cdot \underbrace{\mat(x)}_{m \times 1} + \beta \cdot \underbrace{\mat(y)}_{d \times 1},    \tag{$\star$}
///
/// otherwise, we perform
///
/// .. math::
///     \mat(y) = \alpha \cdot \underbrace{\submat(\mtxS)^T}_{m \times d} \cdot \underbrace{\mat(x)}_{d \times 1} + \beta \cdot \underbrace{\mat(y)}_{m \times 1},    \tag{$\diamond$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`\mtxS` is a sketching operator.
/// 
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What are** :math:`\mat(x)` **and** :math:`\mat(y)` **?**
///     
///       They are vectors of shapes :math:`(\mat(x), L_x \times 1)` and :math:`(\mat(y), L_y \times 1),`
///       where :math:`(L_x, L_y)` are lengths so that :math:`opS(\submat(\mtxS)) \mat(x)` is well-defined and the same shape as :math:`\mat(y).` 
///       Their precise contents are determined in a way that is identical to GEMV from BLAS.
///
///     **Why no "layout" argument?**
///     
///       GEMV in CBLAS accepts a parameter that specifies row-major or column-major layout of the matrix operand.
///       Since our matrix is a sketching operator, and since RandBLAS has no notion of the layout of a sketching operator, we do not have a layout parameter.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      opS - [in]
///       * Either Op::Trans or Op::NoTrans.
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS).`
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)^T.`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\submat(\mtxS).`
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\submat(\mtxS).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`x` is not accessed.
///     
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
///       * Defines :math:`\submat(\mtxS).`
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * :math:`\submat(\mtxS)` is a contiguous submatrix of :math:`S[\texttt{ro_s}:(\texttt{ro_s} + d), :].`
///
///      co_s - [in]
///       * A nonnegative integer.
///       * :math:`\submat(\mtxS)` is a contiguous submatrix of :math:`S[:,\texttt{co_s}:(\texttt{co_s} + m)].`
///
///      x - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(x).`
///
///      incx - [in]
///       * A positive integer.
///       * Stride between elements of x.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`y` need not be set on input.
///
///      y - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(y)` on the RIGHT-hand side of
///         :math:`(\star)` (if :math:`\opS = \texttt{NoTrans}`) or
///         :math:`(\diamond)` (if :math:`\opS = \texttt{Trans}`)
///       * On exit, defines :math:`\mat(y)` on the LEFT-hand side of the same.
///
///      incy - [in]
///       * A positive integer.
///       * Stride between elements of y.
///
/// @endverbatim
template <SketchingOperator SKOP, typename T = SKOP::scalar_t>
inline void sketch_vector(
    blas::Op opS,
    int64_t d, // rows in submat(\mtxS)
    int64_t m, // cols in submat(\mtxS)
    T alpha,
    SKOP &S,
    int64_t ro_s,
    int64_t co_s,
    const T *x,
    int64_t incx,
    T beta,
    T *y,
    int64_t incy
) {
    int64_t _d, _m;
    if (opS == blas::Op::Trans) {
        _d = m;
        _m = d;
    } else {
        _d = d;
        _m = m;
    }
    return sketch_general(blas::Layout::RowMajor, opS, blas::Op::NoTrans, _d, 1, _m, alpha, S, ro_s, co_s, x, incx, beta, y, incy);
}

// MARK: FULL(S)

// =============================================================================
/// \fn sketch_vector(blas::Op opS, T alpha, SKOP &S,
///    const T *x, int64_t incx, T beta, T *y, int64_t incy
/// )
/// @verbatim embed:rst:leading-slashes
/// Perform a GEMV-like operation:
///
/// .. math::
///     \mat(y) = \alpha \cdot \op(\mtxS) \cdot \mat(x) + \beta \cdot \mat(y),    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`\mtxS` is a sketching operator.
/// 
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What are** :math:`\mat(x)` **and** :math:`\mat(y)` **?**
///
///       They are vectors of shapes :math:`(\mat(x), L_x \times 1)` and :math:`(\mat(y), L_y \times 1),`
///       where :math:`(L_x, L_y)` are lengths so that :math:`\opS(\mtxS) \mat(x)` is well-defined and the same shape as :math:`\mat(y).` 
///       Their precise contents are determined in a way that is identical to GEMV from BLAS.
///
///     **Why no "layout" argument?**
///     
///       GEMV in CBLAS accepts a parameter that specifies row-major or column-major layout of the matrix operand.
///       Since our matrix is a sketching operator, and since RandBLAS has no notion of the layout of a sketching operator, we do not have a layout parameter.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      opS - [in]
///       * Either Op::Trans or Op::NoTrans.
///       * If :math:`\opS` = NoTrans, then :math:`\op(\mtxS) = \mtxS.`
///       * If :math:`\opS` = Trans, then :math:`\op(\mtxS) = \mtxS^T.`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`x` is not accessed.
///     
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
///
///      x - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(x).`
///
///      incx - [in]
///       * A positive integer.
///       * Stride between elements of x.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`y` need not be set on input.
///
///      y - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(y)` on the RIGHT-hand side of
///         :math:`(\star).`
///       * On exit, defines :math:`\mat(y)` on the LEFT-hand side of the same.
///
///      incy - [in]
///       * A positive integer.
///       * Stride between elements of y.
///
/// @endverbatim
template <SketchingOperator SKOP, typename T = SKOP::scalar_t>
inline void sketch_vector(
    blas::Op opS,
    T alpha,
    SKOP &S,
    const T *x,
    int64_t incx,
    T beta,
    T *y,
    int64_t incy
) {
    int64_t d = S.dist.n_rows;
    int64_t m = S.dist.n_cols;
    return sketch_vector(opS, d, m, alpha, S, 0, 0, x, incx, beta, y, incy);
}

}  // end namespace RandBLAS
