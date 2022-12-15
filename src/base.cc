#include <RandBLAS/base.hh>
#include <cstring>

namespace RandBLAS::base {

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
    std::memcpy(this->ctr, in_state.ctr.v, this->len_c * 4);
    std::memcpy(this->key, in_state.key.v, this->len_k * 4);
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
    int ctr_len = MIN(this->len_c, s.len_c);
    std::memcpy(this->ctr.v, s.ctr, 4 * ctr_len);
    int key_len = MIN(this->len_k, s.len_k);
    std::memcpy(this->key.v, s.key, 4 * key_len);
}

template bool generator_type_is_same<Philox>(const RNGState &s);
template bool generator_type_is_same<Threefry>(const RNGState &s);

template RNGState::RNGState(const Random123_RNGState<Philox> &in_state);
template RNGState::RNGState(const Random123_RNGState<Threefry> &in_state);

template Random123_RNGState<Philox>::Random123_RNGState(const RNGState &s);
template Random123_RNGState<Threefry>::Random123_RNGState(const RNGState &s);

}
