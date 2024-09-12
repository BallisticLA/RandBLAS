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

#if defined(__GNUC__)
#define RandBLAS_WARNING_COMMENT_OFF _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wcomment\"")
#define RandBLAS_WARNING_COMMENT_ON _Pragma("GCC diagnostic pop")
#else
#define RandBLAS_WARNING_COMMENT_OFF
#define RandBLAS_WARNING_COMMENT_ON
#endif


#ifdef __clang__
#define RandBLAS_OPTIMIZE_OFF _Pragma("clang optimize off")
#define RandBLAS_OPTIMIZE_ON _Pragma("clang optimize on")
#elif defined(__GNUC__)
#define RandBLAS_OPTIMIZE_OFF _Pragma("GCC push_options") _Pragma("GCC optimize (\"O0\")")
#define RandBLAS_OPTIMIZE_ON _Pragma("GCC pop_options")
#elif defined(_MSC_VER)
#define RandBLAS_OPTIMIZE_OFF __pragma(optimize("", off))
#define RandBLAS_OPTIMIZE_ON __pragma(optimize("", on))
#elif defined(__INTEL_COMPILER)
#define RandBLAS_OPTIMIZE_OFF _Pragma("optimize('', off)")
#define RandBLAS_OPTIMIZE_ON _Pragma("optimize('', on)")
#else
#define RandBLAS_OPTIMIZE_OFF
#define RandBLAS_OPTIMIZE_ON
#endif
