#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include <Random123/array.h>
#include <Random123/ReinterpretCtr.hpp>

// enum class GeneratorName : char {Philox = 'P', ThreeFry = 'T'};

struct RNGCounter {
    //uint32_t *v = new uint32_t[4] {0, 0, 0, 0};
    typedef typename r123::Array4x32 ctr_type;
    r123::Array4x32 _v {};
    uint32_t *v = nullptr;

    RNGCounter(uint32_t v0);

    RNGCounter(uint64_t v0);
};

RNGCounter::RNGCounter(uint32_t v0) {
    this->v = this->_v.v;
    this->v[0] = v0;
}

RNGCounter::RNGCounter(uint64_t v0) {
    this->v = this->_v.v;
    this->_v.incr(v0);
}