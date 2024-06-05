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
#include "comparison.hh"
#include "RandBLAS/skge.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <thread>


class TestSketchVector : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template<typename T>
    static void test_sketch_vec_wide(
        uint32_t seed, // Seed for S_wide
        int64_t d,     // Dim of y_actual and y_expect
        int64_t m,     // Dim of x. Expect m > d.
        int64_t incx,  // Stride between elements of x
        int64_t incy   // Stride between elements of y_actual and y_expect.
    ) { 
        T *x        = new T[incx * m];
        T *y_actual = new T[incy * d];
        T *y_expect = new T[incy * d];
        for (int i = 0; i < m; i++)
            x[incx*i] = 1.0;

        RandBLAS::DenseDist D(d, m);
        RandBLAS::DenseSkOp<T> S(D, seed);
        RandBLAS::fill_dense(S);
        int64_t lds = (S.layout == blas::Layout::RowMajor) ? m : d;

        RandBLAS::sketch_vector<T>(blas::Op::NoTrans, d, m, 1.0, S, 0, 0, x, incx, 0.0, y_actual, incy);
        blas::gemv(S.layout, blas::Op::NoTrans, d, m, 1.0, S.buff, lds, x, incx, 0.0, y_expect, incy); 

        test::comparison::buffs_approx_equal(d, y_actual, incy, y_expect, incy,
                __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        delete [] x;
        delete [] y_expect;
        delete [] y_actual;
    }
    
    template<typename T>
    static void test_apply_transposed_to_vector(
        uint32_t seed, // Seed for S_wide
        int64_t d,     // Dim of y
        int64_t m,     // Dim of x. Expect m > d.
        int64_t incx,  // Stride between elements of x
        int64_t incy   // Stride between elements of y_wide and y_ref.
    ) {
        T *x        = new T[incx * m];
        T *y_actual = new T[incy * d];
        T *y_expect = new T[incy * d];
        for (int i = 0; i < m; i++)
            x[incx*i] = 1.0;
        
        RandBLAS::DenseDist D(m, d);
        RandBLAS::DenseSkOp<T> S(D, seed);
        RandBLAS::fill_dense(S);
        int64_t lds = (S.layout == blas::Layout::RowMajor) ? d : m;

        // Perform tall sketch with Op::Trans
        RandBLAS::sketch_vector<T>(blas::Op::Trans, m, d, 1.0, S, 0, 0, x, incx, 0, y_actual, incy);
        blas::gemv(S.layout, blas::Op::Trans, m, d, 1.0, S.buff, lds, x, incx, 0, y_expect, incy); 
        
        // Compare entrywise results of sketching with sketch_vector and using gemv
        test::comparison::buffs_approx_equal(d, y_actual, incy, y_expect, incy,
                __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        delete [] x;
        delete [] y_actual;
        delete [] y_expect;
    }

    template<typename T>
    static void test_transpose_compatible(
        uint32_t seed, // Seed for S_wide
        int64_t d,     // Dim of y
        int64_t m,     // Dim of x. Expect m > d.
        int64_t incx,  // Stride between elements of x
        int64_t incy   // Stride between elements of y_wide and y_tall.
    ) {
        T *x      = new T[incx * m];
        T *y_wide = new T[incy * d];
        T *y_tall = new T[incy * d];
        for (int i = 0; i < m; i++)
            x[i*incx] = 1.0;

        // Generate wide and tall sketching operator using same seed
        RandBLAS::DenseDist D_wide(d, m);
        RandBLAS::DenseDist D_tall(m, d);
        RandBLAS::DenseSkOp<T> S_wide(D_wide, seed);
        RandBLAS::fill_dense(S_wide);
        RandBLAS::DenseSkOp<T> S_tall(D_tall, seed);
        RandBLAS::fill_dense(S_tall);

        // Perform wide sketch with Op::NoTrans and tall sketch with Op::Trans. Should be the same operation
        RandBLAS::sketch_vector<T>(blas::Op::NoTrans, d, m, 1.0, S_wide, 0, 0, x, incx, 0.0, y_wide, incy);
        RandBLAS::sketch_vector<T>(blas::Op::Trans, m, d, 1.0, S_tall, 0, 0, x, incx, 0.0, y_tall, incy);
        
        test::comparison::buffs_approx_equal(d, y_wide, incy, y_tall, incy,
                __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        delete [] x;
        delete [] y_wide;
        delete [] y_tall;
    }

    template<typename T>
    static void test_sketch_vec_tallSK(
        uint32_t seed, // Seed for S_tall
        int64_t d,     // Dim of y
        int64_t m,     // Dim of x. Expect m < d.
        int64_t incx,  // Stride between elements of x
        int64_t incy   // Stride between elements of y_tall and y_ref.
    ) {
        T *x        = new T[incx * m];
        T *y_actual = new T[incy * d];
        T *y_expect = new T[incy * d];
        for (int i = 0; i < m; i++)
            x[incx*i] = 1.0;
        
        // Generate tall sketching operator
        RandBLAS::DenseDist D(d, m);
        RandBLAS::DenseSkOp<T> S(D, seed);
        RandBLAS::fill_dense(S);
        int64_t lds = (S.layout == blas::Layout::RowMajor) ? m : d;

        // Perform tall sketch
        RandBLAS::sketch_vector<T>(blas::Op::NoTrans, d, m, 1, S, 0, 0, x, incx, 0, y_actual, incy);
        blas::gemv(S.layout, blas::Op::NoTrans, d, m, 1, S.buff, lds, x, incx, 0, y_expect, incy); 

        // Compare entrywise results of sketching with sketch_vector and using gemv
        test::comparison::buffs_approx_equal(d, y_actual, incy, y_expect, incy,
                __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        delete [] x;
        delete [] y_actual;
        delete [] y_expect;
    }
};

////////////////////////////////////////////////////////////////////////
//
//
//     Sketching vectors (vary tall vs wide operators) 
//
//
////////////////////////////////////////////////////////////////////////
TEST_F(TestSketchVector, test_sketch_vec_tallSK)
{
    for (uint32_t seed : {0, 1, 2}) {
        test_sketch_vec_tallSK<double>(seed, 1000, 100, 2, 3);
        test_sketch_vec_tallSK<double>(seed, 1013, 101, 3, 2);
    }
}
TEST_F(TestSketchVector, test_transpose_compatible)
{
    for (uint32_t seed : {0, 1, 2}) {
        test_transpose_compatible<double>(seed, 100, 1000, 2, 3);
        test_transpose_compatible<double>(seed, 101, 1013, 3, 2);
    }
}

TEST_F(TestSketchVector, test_sketch_vec_wide)
{
    for (uint32_t seed : {0, 1, 2}) {
        test_sketch_vec_wide<double>(seed, 100, 1000, 2, 3);
        test_sketch_vec_wide<double>(seed, 101, 1013, 3, 2);
    }
}

TEST_F(TestSketchVector, test_apply_transposed_to_vector)
{
    for (uint32_t seed : {0, 1, 2}) {
        test_apply_transposed_to_vector<double>(seed, 100, 1000, 2, 3);
        test_apply_transposed_to_vector<double>(seed, 101, 1013, 3, 2);
    }
}