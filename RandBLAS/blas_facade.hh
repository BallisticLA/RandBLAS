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

#pragma once

/// @file
///
/// BLAS backend selector. Include this header instead of <blas.hh> throughout
/// RandBLAS. The backend is chosen by the following priority chain:
///
///   1. RANDBLAS_BLAS_HEADER (preprocessor macro, highest priority)
///        Define this macro to the quoted path of any header that provides
///        the blas:: namespace (Layout, Op, scal, gemm, …).  Use this to
///        supply a CBLAS wrapper (see blas_backend/cblas.hh) or your own
///        implementation without touching CMake:
///          -DRANDBLAS_BLAS_HEADER='"RandBLAS/blas_backend/cblas.hh"'
///
///   2. RandBLAS_HAS_BLASPP (set by CMake when BLAS++ is found)
///        Uses BLAS++ (<blas.hh>).
///
///   3. Builtin fallback (CMake option RandBLAS_BLAS_BACKEND=builtin)
///        Uses RandBLAS's own portable implementations.
///
/// All three paths expose the same blas::Layout, blas::Op, blas::scal, etc.

#include "RandBLAS/config.h"

#if defined(RANDBLAS_BLAS_HEADER)
    #include RANDBLAS_BLAS_HEADER
#elif defined(RandBLAS_HAS_BLASPP)
    #include <blas.hh>
#else
    #include "RandBLAS/blas_backend/builtin.hh"
#endif
