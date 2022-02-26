#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RBLAS_SJLTS_HH
#define RBLAS_SJLTS_HH

namespace rblas::sjlts {

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

void fill_colwise(SJLT sjl, uint64_t seed_key, uint64_t seed_ctr);

void sketch_cscrow(SJLT sjl, uint64_t n, double *a, double *a_hat, int threads);

void print_sjlt(SJLT sjl);

} // end namespace rblas::sjlts

#endif // define RBLAS_SJLTS_HH
