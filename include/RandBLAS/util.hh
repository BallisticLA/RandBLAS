#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_UTIL_HH
#define RandBLAS_UTIL_HH

namespace RandBLAS::util {


void genmat(
	int64_t n_rows,
	int64_t n_cols,
	double* mat,
	uint64_t seed);


//template<typename uint64_t, typename TA>
//void print_colmaj(uint64_t n_rows, uint64_t n_cols, double *a, char label[]);

void print_colmaj(uint64_t n_rows, uint64_t n_cols, double *a, char label[]);

} // end namespace RandBLAS::util

#endif  // define RandBLAS_UTIL_HH
