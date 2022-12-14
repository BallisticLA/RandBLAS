#include "state.hh"


RNGState::RNGState(
    uint32_t c0,
    uint32_t k0
) : len_c(4),
    len_k(4)
{
    this->ctr = new uint32_t[this->len_c];
    this->key = new uint32_t[this->len_k];
    this->ctr[0] = c0;
    this->key[0] = k0;
}

RNGState::RNGState(
    const RNGState &s
) : len_c(s.len_c),
    len_k(s.len_k)
{
    this->ctr = new uint32_t[this->len_c];
    this->key = new uint32_t[this->len_k];
    std::memcpy(this->ctr, s.ctr, this->len_c * 4);
    std::memcpy(this->key, s.key, this->len_k * 4);
}

// Convert from _R123State_ to RNGState
template <typename T_state>
RNGState::RNGState(
    const T_state &in_state
) : len_c(in_state.len_c),
    len_k(in_state.len_k)
{
    this->ctr = new uint32_t[this->len_c];
    this->key = new uint32_t[this->len_k];
    std::memcpy(this->ctr, in_state.ctr.v, this->len_c * 4);
    std::memcpy(this->key, in_state.key.v, this->len_k * 4);
}

// convert from RNGState to _R123State_
template <typename T_gen>
_R123State_<T_gen>::_R123State_(
    const RNGState &s
) : ctr{},
    key{},
    len_c(T_gen::ctr_type::static_size),
    len_k(T_gen::key_type::static_size) 
{
    int ctr_len = (this->len_c <= s.len_c) ? this->len_c : s.len_c; 
    std::memcpy(this->ctr.v, s.ctr, 4 * ctr_len);
    int key_len = (this->len_k <= s.len_k) ? this->len_k : s.len_k; 
    std::memcpy(this->key.v, s.key, 4 * key_len);
}

//template RNGState::RNGState(uint32_t c0, uint32_t k0);
//template RNGState::RNGState(const RNGState &s);

template RNGState::RNGState(const _R123State_<Philox4x32> &in_state);
template RNGState::RNGState(const _R123State_<Threefry4x32> &in_state);

template _R123State_<Philox4x32>::_R123State_(const RNGState &s);
template _R123State_<Threefry4x32>::_R123State_(const RNGState &s);
