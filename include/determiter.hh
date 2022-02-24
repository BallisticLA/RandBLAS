//#include <vector>

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

template <typename T>
void pcg(
	int64_t m,
	int64_t n,
	T* const A,
	int64_t lda, 
	T* const b, // length m
	T* const c, // length n
    T delta, // >= 0
	std::vector<T>& resid_vec, // re
	T tol, //  > 0
	int64_t k,
	T* const M, // n-by-k
	int64_t ldm,
	T* const x0, // length n
	T* x,  // length n
	T* y // length m
);

void run_pcgls_ex(int64_t n, int64_t m);
