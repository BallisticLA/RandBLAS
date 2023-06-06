#include <RandBLAS/base.hh>
#include <cstring>

namespace RandBLAS {

RNGState::RNGState(
    uint32_t k0
) : len_c(4),
    len_k(4)
{
    this->ctr = new uint32_t[this->len_c]{};
    this->key = new uint32_t[this->len_k]{};
    this->key[0] = k0;
}

RNGState::~RNGState()
{
    delete [] this->ctr;
    delete [] this->key;
}

RNGState::RNGState(
    const RNGState &s
) : len_c(s.len_c),
    len_k(s.len_k)
{
    this->ctr = new uint32_t[this->len_c];
    this->key = new uint32_t[this->len_k];
    std::memcpy(this->ctr, s.ctr, this->len_c * sizeof(uint32_t));
    std::memcpy(this->key, s.key, this->len_k * sizeof(uint32_t));
}

RNGState::RNGState(
    RNGState &&s
)
{
    std::swap(this->len_c, s.len_c);
    std::swap(this->len_k, s.len_k);
    std::swap(this->ctr, s.ctr);
    std::swap(this->key, s.key);
}

RNGState &RNGState::operator=(
    const RNGState &s
)
{
    delete [] this->ctr;
    delete [] this->key;

    this->len_c = s.len_c;
    this->len_k = s.len_k;

    this->ctr = new uint32_t[this->len_c];
    this->key = new uint32_t[this->len_k];

    std::memcpy(this->ctr, s.ctr, this->len_c * sizeof(uint32_t));
    std::memcpy(this->key, s.key, this->len_k * sizeof(uint32_t));

    return *this;
}

RNGState &RNGState::operator=(
    RNGState &&s
)
{
    std::swap(this->len_c, s.len_c);
    std::swap(this->len_k, s.len_k);
    std::swap(this->ctr, s.ctr);
    std::swap(this->key, s.key);
    return *this;
}

std::ostream &operator<<(
    std::ostream &out,
    const RNGState &s
) {
    int i;
    out << "counter : {";
    for (i = 0; i < s.len_c - 1; ++i) {
        out << s.ctr[i] << ", ";
    }
    out << s.ctr[i] << "}\n";
    out << "key     : {";
    for (i = 0; i < s.len_k - 1; ++i) {
        out << s.key[i] << ", ";
    }
    out << s.key[i] << "}";
    return out;
}

}
