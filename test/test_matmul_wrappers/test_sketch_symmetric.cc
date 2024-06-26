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

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/skge.hh"

#include "test/comparison.hh"

#include <gtest/gtest.h>


class TestSketchSymmetric : public ::testing::Test {

    /* TODO
    test_left_apply_to_random
    test_left_apply_submatrix_to_eye
    test_left_apply_transpose_to_eye ---> not applicable
    test_left_apply_to_submatrix
    test_left_apply_to_transposed    ---> not applicable
    */

    /* APPROACH
    Remember, the tests I write now might be silly, but they might
    be nontrivial if I change the implementation of sketch_symmetric.

    
    */


    // template <typename T>
    // static void test_full_skop_from_left(
    //     blas::Layout layout,
    //     int64_t n,
    //     int64_t d,
    // ) {

    // }

    // template <typename T>
    // static void test_full_skop_from_right(

    // ) {

    // }

    // template <typename T>
    // static void test_submat_from_left(

    // ) {

    // }

    // template <typename T>
    // static void test_submat_from_right(

    // ) {

    // }

};