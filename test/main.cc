#include <rblas/util.hh>
#include <determiter.hh>
#include <rblas/sjlts.hh>
#include <blas.hh>
#include "gtest/gtest.h"


int main(int argc, char *argv[]) {
    
    /*uint64_t m = 15;
    uint64_t n = 4;
    uint64_t a_seed = 42;
    uint64_t s_seed = 15;

    double *a = new double[m * n];
    rblas::util::genmat(m, n, a, a_seed);
    rblas::util::print_colmaj(n, m, a, "transpose(a)");

    struct rblas::sjlts::SJLT sjl;
    sjl.ori = rblas::sjlts::ColumnWise;
    sjl.n_rows = 8; // > n
    sjl.n_cols = m;
    sjl.vec_nnz = 3; // <= n_rows
    uint64_t *rows = new uint64_t[sjl.vec_nnz * sjl.n_cols];
    sjl.rows = rows;
    uint64_t *cols = new uint64_t[sjl.vec_nnz * sjl.n_cols];
    sjl.cols = cols;
    double *vals = new double[sjl.vec_nnz * sjl.n_cols];
    sjl.vals = vals;
    rblas::sjlts::fill_colwise(sjl, s_seed, 0);
    double *a_hat = new double[sjl.n_rows * n];
    for (int i = 0; i < sjl.n_rows * n; ++i)
    {
        a_hat[i] = 0.0;
    }
    rblas::sjlts::sketch_cscrow(sjl, n, a, a_hat);
    
    rblas::util::print_colmaj(n, sjl.n_rows, a_hat, "tranpose(a_hat) -- Tianyu");

    rblas::sjlts::print_sjlt(sjl);
    */

    ::testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();
    return res;
}
