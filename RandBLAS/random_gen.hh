#ifndef random_gen_hh
#define random_gen_hh

// this is for sincosf
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <Random123/array.h>
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/aes.h>
#include <Random123/ars.h>
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>

/// @cond
/// extend r123::float2 to work with structured bindings
namespace std {
  template<> struct tuple_size<r123::float2> { static constexpr size_t value = 2; };
  template<> struct tuple_element<0, r123::float2> { using type = float; };
  template<> struct tuple_element<1, r123::float2> { using type = float; };
}

namespace r123 {
template<std::size_t I>
std::tuple_element_t<I, r123::float2> get(r123::float2 const& f2)
{
  if constexpr (I == 0) return f2.x;
  if constexpr (I == 1) return f2.y;
}
}
/// @endcond

#endif
