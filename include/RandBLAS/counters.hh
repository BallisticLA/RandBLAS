#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include <Random123/array.h>
#include <Random123/ReinterpretCtr.hpp>

// enum class GeneratorName : char {Philox = 'P', ThreeFry = 'T'};

struct RNGState {
    uint32_t *c = nullptr;
    uint32_t *k = nullptr;
    typedef typename r123::Array4x32 r123_ctr;
    r123::Array4x32 _c {};
    r123::Array2x32 _k {};

    RNGState(uint32_t c0, uint32_t k0);

    RNGState(uint64_t c0, uint32_t k0);
};

RNGState::RNGState(uint32_t c0, uint32_t k0) {
    this->c = this->_c.v;
    this->k = this->_k.v;
    this->c[0] = c0;
    this->k[0] = k0;
}

RNGState::RNGState(uint64_t c0, uint32_t k0) {
    this->c = this->_c.v;
    this->k = this->_k.v;
    this->_c.incr(c0);
    this->k[0] = k0;
}