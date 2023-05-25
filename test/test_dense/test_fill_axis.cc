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
        static inline int64_t short_dim = 4;
        static inline int64_t long_dim  = 8;
        static inline auto distname = RandBLAS::dense::DenseDistName::Uniform;

    template <typename T>
    static void auto_transpose(blas::Layout layout, RandBLAS::dense::FillAxis fa) {
        uint32_t seed = 99;
        // make the wide sketching operator
        RandBLAS::dense::DenseDist D_wide {short_dim, long_dim, distname, fa};
        RandBLAS::dense::DenseSkOp<T> S_wide(D_wide, seed, nullptr, layout);
        RandBLAS::dense::realize_full(S_wide);  
        // make the tall sketching operator
        RandBLAS::dense::DenseDist D_tall {long_dim, short_dim, distname, fa};
        RandBLAS::dense::DenseSkOp<T> S_tall(D_tall, seed, nullptr, layout);
        RandBLAS::dense::realize_full(S_tall);
        // check that buffers reflect transposed data
        auto lds_wide = (layout == blas::Layout::ColMajor) ? short_dim : long_dim;
        auto lds_tall = (layout == blas::Layout::ColMajor) ? long_dim  : short_dim;
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, blas::Op::Trans, short_dim, long_dim,
            S_wide.buff, lds_wide, S_tall.buff, lds_tall,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        // Tests don't pass .... problem :
        //      If fa==Long (resp. Short), then we can only hope to get correct
        //      results by reading S_wide in row-major (resp. col-major) 
        //                     and S_tall in col-major (resp. row-major).
        //
        // So, looks like FillAxis and shape need to determine row-major vs column-major.
        //
        // Problem: what if you want to compute S_wide * A where A is in column-major format
        // and S_wide.fill_axis==Long? If I go with the observation above about row-major
        // and column-major, then S_wide with fill_axis==Long would need to be in row-major.
        // But the current implementation of sketch_gefx3 requires that S_wide be in the same
        // format as A (i.e., column-major). Need to figure this out ...
        //
        //      Task: compute S_wide*A in column major where A is in column-major and S_wide
        //      is in row-major. How about, compute Trans(mat(S_wide, as_col_major)) * S?
        return;   
    }

};

TEST_F(TestFillAxis, long_axis_col_major) {
    auto_transpose<float>(blas::Layout::ColMajor, RandBLAS::dense::FillAxis::Long);
}

TEST_F(TestFillAxis, long_axis_row_major) {
    auto_transpose<float>(blas::Layout::RowMajor, RandBLAS::dense::FillAxis::Long);
}

TEST_F(TestFillAxis, short_axis_col_major) {
    auto_transpose<float>(blas::Layout::ColMajor, RandBLAS::dense::FillAxis::Short);
}

TEST_F(TestFillAxis, short_axis_row_major) {
    auto_transpose<float>(blas::Layout::RowMajor, RandBLAS::dense::FillAxis::Short);
}
