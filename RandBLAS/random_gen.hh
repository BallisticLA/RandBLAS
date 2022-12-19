#ifndef random_gen_hh
#define random_gen_hh

// this is for sincosf
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <Random123/array.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/boxmuller.hpp>

using Philox = r123::Philox4x32;
using Threefry = r123::Threefry4x32;

#endif
