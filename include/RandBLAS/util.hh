#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_UTIL_HH
#define RandBLAS_UTIL_HH

namespace RandBLAS::util {

template <typename T>
void genmat(
	int64_t n_rows,
	int64_t n_cols,
	T* mat,
	uint64_t seed
);

template <typename T>
void print_colmaj(
	int64_t n_rows,
	int64_t n_cols,
	T *a,
	char label[]
);

} // end namespace RandBLAS::util

#endif  // define RandBLAS_UTIL_HH
