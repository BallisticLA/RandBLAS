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

#ifndef randblas_sksy_hh
#define randblas_sksy_hh

#include "RandBLAS/util.hh"
#include "RandBLAS/base.hh"
#include "RandBLAS/skge.hh"

namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;

/* Intended macro definitions.

   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |lda| mathmacro:: \texttt{lda}
   .. |ldb| mathmacro:: \texttt{ldb}
   .. |opA| mathmacro:: \texttt{opA}
   .. |opS| mathmacro:: \texttt{opS}
*/

template <typename T, typename SKOP>
inline void sketch_symmetric(
    // B = alpha*A*S + beta*B, where A is a symmetric matrix stored in the format of a general matrix.
    blas::Layout layout, // layout for (A,B)
    int64_t n, // number of rows in B
    int64_t d, // number of columns in B
    T alpha,
    const T* A,
    int64_t lda,
    SKOP &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T* B,
    int64_t ldb,
    T sym_check_tol = 0
) {
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, n, d, n, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
}

template <typename T, typename SKOP>
inline void sketch_symmetric(
    // B = alpha*S*A + beta*B
    blas::Layout layout, // layout for (A,B)
    int64_t d, // number of rows in B
    int64_t n, // number of columns in B
    T alpha,
    SKOP &S,
    int64_t ro_s,
    int64_t co_s,
    const T* A,
    int64_t lda,
    T beta,
    T* B,
    int64_t ldb,
    T sym_check_tol = 0
) {
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, d, n, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
}

} // end namespace RandBLAS
#endif
