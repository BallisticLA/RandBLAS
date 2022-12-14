#include "state.hh"


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

RNGState::RNGState(const RNGState &s) {
    this->k = this->_k.v;
    this->c = this->_c.v;
    std::memcpy(this->c, s.c, 16);  // 4 x 4-byte floats
    std::memcpy(this->k, s.k, 8);
}
