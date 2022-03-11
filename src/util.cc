#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>


namespace RandBLAS::util {

void genmat(
	int64_t n_rows,
	int64_t n_cols,
	double* mat,
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
		mat[i] = r123::uneg11<double>(rand.v[0]);
	}
}


void print_colmaj(uint64_t n_rows, uint64_t n_cols, double *a, char label[])
{
	uint64_t i, j;
    double val;
	std::cout << "\n" << label << std::endl;
    for (i = 0; i < n_rows; ++i) {
        std::cout << "\t";
        for (j = 0; j < n_cols - 1; ++j) {
            val = a[i + n_rows * j];
            if (val < 0) {
				//std::cout << string_format("  %2.4f,", val);
                printf("  %2.4f,", val);
            } else {
				//std::cout << string_format("   %2.4f", val);
				printf("   %2.4f,", val);
            }
        }
        // j = n_cols - 1
        val = a[i + n_rows * j];
        if (val < 0) {
   			//std::cout << string_format("  %2.4f,", val); 
			printf("  %2.4f,", val);
		} else {
            //std::cout << string_format("   %2.4f,", val);
			printf("   %2.4f,", val);
		}
        printf("\n");
    }
    printf("\n");
    return;
}


} // end namespace RandBLAS::util
