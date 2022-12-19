#ifndef randblas_base_hh
#define randblas_base_hh

#include "RandBLAS/random_gen.hh"

#include <typeinfo>
#include <typeindex>
#include <cstring>

#include<iostream>

namespace RandBLAS::base {

enum class RNGName : char {None = '\0', Philox = 'P', Threefry = 'T'};

struct RNGState {

    int len_c = 0;
    int len_k = 0;
    uint32_t *ctr = nullptr;
    uint32_t *key = nullptr;
    RNGName rng_name = RNGName::Philox; // TODO -- use None here

    RNGState() : len_c(0), len_k(0), ctr(nullptr), key(nullptr), rng_name(RNGName::None)  {};

    ~RNGState();

    RNGState(const RNGState &s);
    RNGState(RNGState &&s);

    RNGState(uint32_t c0, uint32_t k0);

    template <typename T_state>
    RNGState(const T_state &in_state);

    RNGState &operator=(const RNGState &s);
    RNGState &operator=(RNGState &&s);
};

std::ostream &operator<<(std::ostream &out, const RNGState &s);

template <typename T_gen>
struct Random123_RNGState {

    typedef T_gen gen_type;
    typedef typename T_gen::key_type key_type;
    typedef typename T_gen::ctr_type ctr_type;

    ctr_type ctr{};
    key_type key{};

    const int len_c = ctr_type::static_size;
    const int len_k = key_type::static_size;

    Random123_RNGState(const RNGState &s);
};

// Convert from Random123_RNGState to RNGState
template <typename T_state>
RNGState::RNGState(
    const T_state &in_state
) : len_c(in_state.len_c),
    len_k(in_state.len_k)
{
    typedef typename T_state::gen_type gen_type;
    Philox ph;
    Threefry tf;
    gen_type gt;
    auto gtid = ::std::type_index(typeid(gt));
    if (gtid == ::std::type_index(typeid(ph))) {
        this->rng_name = RNGName::Philox;
    } else if (gtid == ::std::type_index(typeid(tf))) {
        this->rng_name = RNGName::Threefry;
    } else {
        throw std::runtime_error(std::string("Unknown gen_type."));
    }

    this->ctr = new uint32_t[this->len_c];
    this->key = new uint32_t[this->len_k];

    memcpy(this->ctr, in_state.ctr.v, this->len_c * sizeof(uint32_t));
    memcpy(this->key, in_state.key.v, this->len_k * sizeof(uint32_t));
}

template <typename T_gen>
bool generator_type_is_same(
    const RNGState &s
) {
    T_gen gt;
    auto gtid = ::std::type_index(typeid(gt));
    switch (s.rng_name) {
        case RNGName::Philox: {
              Philox ph;
              auto phid = ::std::type_index(typeid(ph));
              return (phid == gtid);  
        }
        case RNGName::Threefry: {
            Threefry tf;
            auto tfid = ::std::type_index(typeid(tf));
            return (tfid == gtid);
        }
        default:
            throw std::runtime_error(std::string("Unrecognized rng_name."));
    }
}

// convert from RNGState to Random123_RNGState
template <typename T_gen>
Random123_RNGState<T_gen>::Random123_RNGState(
    const RNGState &s
) : ctr{},
    key{},
    len_c(T_gen::ctr_type::static_size),
    len_k(T_gen::key_type::static_size)
{
    bool res = generator_type_is_same<T_gen>(s);

    if (!res) {
        throw std::runtime_error(std::string("T_gen must match s.rng_name."));
    }

    int ctr_len = std::min(this->len_c, s.len_c);
    memcpy(this->ctr.v, s.ctr, sizeof(uint32_t) * ctr_len);

    int key_len = std::max(this->len_k, s.len_k);
    memcpy(this->key.v, s.key, sizeof(uint32_t) * key_len);
}

}; // end namespace RandBLAS::base

#endif
