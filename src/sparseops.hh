#include <iostream>
#include <blas.hh>
#include <vector>
#include <stdio.h>
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

enum sjlt_orientation {ColumnWise, RowWise};

struct SJLT {
    sjlt_orientation ori;
    uint64_t n_rows;
    uint64_t n_cols;
    uint64_t vec_nnz;
    uint64_t *rows;
    uint64_t *cols;
    double *vals;
};

void fill_colwise_sjlt(SJLT sjl, uint64_t seed_key, uint64_t seed_ctr);

void print_sjlt(SJLT sjl);
