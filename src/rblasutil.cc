#include <iostream>
#include <blas.hh>
#include <stdio.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>


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
	std::cout << std::hex << "The first few 2x64 randoms from Philox2x64 with hex key " << key << std::endl;
	double x, y;
	for (int i = 0; i < 10; ++i)
	{
		ctr[0] = i;
		CBRNG::ctr_type rand = g(ctr, key);
		x = r123::uneg11<double>(rand.v[0]);
		y = r123::uneg11<double>(rand.v[1]);
		std::cout << "ctr: " << ctr << " Philox4x64<>(ctr,key): " << x << ' ' << y << std::endl;
	}
}

