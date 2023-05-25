#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"
#include "RandBLAS/ramm.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <thread>



class TestFillAxis : public::testing::Test
{
    protected:
        static inline auto distname = RandBLAS::dense::DenseDistName::Uniform;

    template <typename T>
    static void auto_transpose(int64_t short_dim, int64_t long_dim, RandBLAS::dense::FillAxis fa) {
        uint32_t seed = 99;
    
        // make the wide sketching operator
        RandBLAS::dense::DenseDist D_wide {short_dim, long_dim, distname, fa};
        RandBLAS::dense::DenseSkOp<T> S_wide(D_wide, seed);
        RandBLAS::dense::realize_full(S_wide);

        // make the tall sketching operator
        RandBLAS::dense::DenseDist D_tall {long_dim, short_dim, distname, fa};
        RandBLAS::dense::DenseSkOp<T> S_tall(D_tall, seed);
        RandBLAS::dense::realize_full(S_tall);

        // Sanity check: layouts are opposite.
        if (S_tall.layout == S_wide.layout) {
            FAIL() << "\n\tExpected opposite layouts.\n";
        }

        // check that buffers reflect transposed data : S_wide == S_tall.T
        auto lds_wide = (S_wide.layout == blas::Layout::ColMajor) ? short_dim : long_dim;
        auto lds_tall = (S_tall.layout == blas::Layout::ColMajor) ? long_dim  : short_dim;
        RandBLAS_Testing::Util::matrices_approx_equal(
            S_wide.layout, S_tall.layout, blas::Op::Trans, short_dim, long_dim,
            S_wide.buff, lds_wide, S_tall.buff, lds_tall,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;   
    }

};

TEST_F(TestFillAxis, long_axis_3x5) {
    auto_transpose<float>(3, 5, RandBLAS::dense::FillAxis::Long);
}

TEST_F(TestFillAxis, short_axis_3x5) {
    auto_transpose<float>(3, 5, RandBLAS::dense::FillAxis::Short);
}

TEST_F(TestFillAxis, long_axis_4x8) {
    auto_transpose<float>(4, 8, RandBLAS::dense::FillAxis::Long);
}

TEST_F(TestFillAxis, short_axis_4x8) {
    auto_transpose<float>(4, 8, RandBLAS::dense::FillAxis::Short);
}

TEST_F(TestFillAxis, long_axis_2x4) {
    auto_transpose<float>(2, 4, RandBLAS::dense::FillAxis::Long);
}

TEST_F(TestFillAxis, short_axis_2x4) {
    auto_transpose<float>(2, 4, RandBLAS::dense::FillAxis::Short);
}

