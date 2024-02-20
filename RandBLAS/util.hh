#ifndef randblas_util_hh
#define randblas_util_hh

#include <blas.hh>
#include <cstdio>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>


namespace RandBLAS::util {

template <typename T>
void increase_stride(T* buff, int64_t cur_stride, int64_t new_stride, int64_t num_vecs) {
    // The first (num_vecs * cur_stride) entries of buff define "num_vecs" vectors of
    // length "cur_stride".
    //
    // buff has at least (num_vecs * new_stride) space.
    //
    // We'll move data farther down buff so each of the num_vecs vectors has
    //      shift = new_stride - cur_stride
    // zeros after it (shift >= 0).
    //
    int64_t incr = new_stride - cur_stride;
    randblas_require(incr >= 0);
    T* work = new T[cur_stride];
    for (int64_t i = num_vecs - 1; i >= 1; --i) {
        // copy the current vector into temporary workspace
        T* vec_tip_before = &buff[i*cur_stride];
        blas::copy(cur_stride, vec_tip_before, 1, work, 1);
        // copy the workspace into the new position of the vector
        T* vec_tip_after = vec_tip_before + i*incr;
        blas::copy(cur_stride, work, 1, vec_tip_after, 1);
    }
    delete [] work;
    // zero out "incr" entries after each vector.
    T* vec_tail = &buff[cur_stride];
    for (int64_t i = 0; i < num_vecs; ++i) {
        for (int64_t j = 0; j < incr; ++j)
            vec_tail[j] = 0.0;
        vec_tail = vec_tail + new_stride;
    }
}

template <typename T>
void safe_scal(int64_t n, T a, T* x, int64_t inc_x) {
    if (a == 0.0) {
        for (int64_t i = 0; i < n; ++i)
            x[i*inc_x] = 0.0;
    } else {
        blas::scal(n, a, x, inc_x);
    }
}

template <typename T>
void genmat(
	int64_t n_rows,
	int64_t n_cols,
	T* mat,
	uint64_t seed)
{
	typedef r123::Philox2x64 CBRNG;
	CBRNG::key_type key = {{seed}};
	CBRNG::ctr_type ctr = {{0,0}};
	CBRNG g;
	uint64_t prod = n_rows * n_cols;
	for (uint64_t i = 0; i < prod; ++i)
	{
		ctr[0] = i;
		CBRNG::ctr_type rand = g(ctr, key);
		mat[i] = r123::uneg11<T>(rand.v[0]);
	}
}

template <typename T>
void print_colmaj(int64_t n_rows, int64_t n_cols, T *a, char label[])
{
	int64_t i, j;
    T val;
	std::cout << "\n" << label << std::endl;
    for (i = 0; i < n_rows; ++i) {
        std::cout << "\t";
        for (j = 0; j < n_cols - 1; ++j) {
            val = a[i + n_rows * j];
            if (val < 0) {
				//std::cout << string_format("  %2.4f,", val);
                printf("  %2.20f,", val);
            } else {
				//std::cout << string_format("   %2.4f", val);
				printf("   %2.20f,", val);
            }
        }
        // j = n_cols - 1
        val = a[i + n_rows * j];
        if (val < 0) {
   			//std::cout << string_format("  %2.4f,", val); 
			printf("  %2.20f,", val);
		} else {
            //std::cout << string_format("   %2.4f,", val);
			printf("   %2.20f,", val);
		}
        printf("\n");
    }
    printf("\n");
    return;
}

template<typename RNG>
bool compare_ctr(typename RNG::ctr_type c1, typename RNG::ctr_type c2) {
    int len = c1.size();
    
    for (int ind = len - 1; ind >= 0; ind--) {
        if (c1[ind] > c2[ind]) {
            return true;
        } else if (c1[ind] < c2[ind]) {
            return false;
        }
    }
    return false;
}


} // end namespace RandBLAS::util

#endif
