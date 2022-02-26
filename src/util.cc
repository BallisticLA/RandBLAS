#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>


namespace rblas::util {

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

// copied from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
/*template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}*/


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


} // end namespace