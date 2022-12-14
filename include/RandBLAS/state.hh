#pragma once

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_STATE_HH
#define RandBLAS_STATE_HH

#include <Random123/array.h>
//#include <Random123/ReinterpretCtr.hpp>

struct RNGState {
    int len_c = 0;
    int len_k = 0;
    uint32_t *ctr = nullptr;
    uint32_t *key = nullptr;

    RNGState() {};

    RNGState(const RNGState &s);

    RNGState(uint32_t c0, uint32_t k0);

    template <typename T_state>
    RNGState(const T_state &in_state);
};

template <typename T_gen>
struct _R123State_ {
    typedef typename T_gen::key_type key_type;
    typedef typename T_gen::ctr_type ctr_type;
    ctr_type ctr{};
    key_type key{};
    const int len_c = ctr_type::static_size;
    const int len_k = key_type::static_size;

    _R123State_(const RNGState &s);
};

// Get access to Philox4x32:
//      Need to patch some missing function definitions.
//      This would pollute RandBLAS namespace, so
//      first we define a private namespace and then
//      refer to the desired types later on.
namespace RandBLAS::__philox__ {

#if !defined(R123_NO_SINCOS) && defined(__APPLE__)
    /* MacOS X 10.10.5 (2015) doesn't have sincosf */
    // use "-D __APPLE__" as a compiler flag to make sure this is hit.
    #define R123_NO_SINCOS 1
#endif

#if R123_NO_SINCOS
    // sincos isn't in the math library
    #include <math.h>
    static inline void sincosf(float x, float *s, float *c) {
        *s = sinf(x);
        *c = cosf(x);
    }
    static inline void sincos(double x, double *s, double *c) {
        *s = sin(x);
        *c = cos(x);
    }
#endif

// Random123 relies on them in both host and device sources.  We can work
// around this by defining them when not compiling device code.
#if !defined(__CUDACC__)
    // The following two functions are part of NVIDIA device side math library.
    static inline void sincospif(float x, float *s, float *c) {
        const float PIf = 3.1415926535897932f;
        sincosf(PIf*x, s, c);
    }

    static inline void sincospi(double x, double *s, double *c) {
        const double PI = 3.1415926535897932;
        sincos(PI*x, s, c);
    }
#endif // Not compiling with CUDA

#include <Random123/philox.h>
}; // end namespace RandBLAS::__philox__


// Get access to ThreeFry4x32
//      This is much simpler than Philox4x32.
namespace RandBLAS::__threefry__ {
#include <Random123/threefry.h>
};


typedef RandBLAS::__philox__::r123::Philox4x32 Philox4x32;
typedef RandBLAS::__threefry__::r123::Threefry4x32 Threefry4x32;

#endif
