/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <cmath>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <charconv>
#include <cassert>
#include <system_error>
#include <cstdint>
#include <map>
#include <string>

#include <cstring>
#include <utility>
#include <stdexcept>

#include <RandBLAS/base.hh>
#include <RandBLAS/random_gen.hh>

#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>
#include <Random123/features/compilerfeatures.h>
#include <Random123/MicroURNG.hpp>
#include <Random123/conventional/Engine.hpp>


#define LINESIZE 1024
#ifdef _MSC_FULL_VER
#define strtoull _strtoui64
// ^ Needed to define the strtou32 and strtou64 functions.
#pragma warning (disable : 4521)
// ^ Engines have multiple copy constructors, quite legal C++, disable MSVC complaint
#endif

int verbose = 0;
int debug = 0;

// MARK: I/O and conversions

/* strdup may or may not be in string.h, depending on the value
   of the pp-symbol _XOPEN_SOURCE and other arcana.  Just
   do it ourselves.
   Mnemonic:  "ntcs" = "nul-terminated character string" */
char *ntcsdup(const char *s){
    char *p = (char *)malloc(strlen(s)+1);
    strcpy(p, s);
    return p;
}

// Functions to read a (portion of) a string in a given base and convert it to
// an unsigned integer.
//
// These functions differ from std::from_chars in how they handle white space.
// Specifically, they strip leading whitespace, and then they stop reading as
// soon as they reach a non-numeric character. (Note that the "a" in 257a3673
// counts as a numeric character if we're reading in hexadecimal format.)
uint32_t strtou32(const char *p, char **endp, int base){
    uint32_t ret;
    errno = 0;
    ret = strtoul(p, endp, base);
    assert(errno==0);
    return ret;
}
uint64_t strtou64(const char *p, char **endp, int base){
    uint64_t ret;
    errno = 0;
    ret = strtoull(p, endp, base);
    assert(errno==0);
    return ret;
}

// A helper function to print unsigned integers in hexadecimal format, with leading zeros if necessary.
template <typename T>
void prtu(std::ostream& os, T val) {
    os << std::hex << std::setw(std::numeric_limits<T>::digits / 4) << std::setfill('0') << val;
    assert(!os.bad());
}
void prtu32(std::ostream& os, uint32_t v) { prtu(os, v); }
void prtu64(std::ostream& os, uint64_t v) { prtu(os, v); }

#define PRINTARRAY(ARR, fp) \
do { \
    char ofmt[64]; \
    size_t xj; \
    /* use %lu and the cast (instead of z) for portability to Microsoft, sizeof(v[0]) should fit easily in an unsigned long.  Avoid inttypes for the same reason. */ \
    snprintf(ofmt, sizeof(ofmt), " %%0%lullx", (unsigned long)sizeof(ARR.v[0])*2UL); \
    for (xj = 0; xj < sizeof(ARR.v)/sizeof(ARR.v[0]); xj++) { \
	fprintf(fp, ofmt, (unsigned long long) ARR.v[xj]); \
    } \
} while(0)

#define PRINTLINE(NAME, N, W, R, ictr, ukey, octr, fp) \
do { \
    fprintf(fp, "%s %d ", #NAME #N "x" #W, R); \
    PRINTARRAY(ictr, fp); \
    putc(' ', fp); \
    PRINTARRAY(ukey, fp); \
    putc(' ', fp); \
    PRINTARRAY(octr, fp); \
    putc('\n', fp); \
    fflush(fp); \
} while(0)

// MARK: Base generator test
//
// There's a lot of code involved in this test. The code can roughly
// be broken down into three categories.
//
// Category 1: code generated from compiler directives
//
//   This pattern is left over from our adaptation of Random123 tests,
//   which have to compile whether interpreted as C or C++ source. It
//   uses compiler directives to accomplish what something roughly 
//   equivalent to C++ templating and metaprogramming.
//
//   The code specifically generates the following identifiers.
//
//      method_e::<GEN>NxW_e        (enum members)
//      <GEN>NxW_kat                (structs)
//      kat_instance.<GEN>NxW_data  (members of type <GEN>NxW_kat)
//      read_<GEN>NxW               (functions)
//      report_<GEN>NxWerror        (functions)
//
// Category 2: helper functions
//
//   The base_rng_test_[arrange,act,assert] functions are slight adaptations
//   of functions that appeared in Random123 testing infrastructure. Their 
//   names indicte their roles in the common "arrange, act, assert" pattern of
//   writing unit tests. Their precise descriptions are complicated.
//   
// Category 3: the main runner
//
//   This manages all calls to the helper functions defined in Category 2.
//

enum method_e{
#define RNGNxW_TPL(base, N, W) base##N##x##W##_e,
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL
    last
};

#define RNGNxW_TPL(base, N, W)                       \
    struct base##N##x##W##_kat {                     \
        base##N##x##W##_ctr_t ctr;                   \
        base##N##x##W##_ukey_t ukey;                 \
        base##N##x##W##_ctr_t expected;              \
        base##N##x##W##_ctr_t computed;              \
    };
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

struct kat_instance {
    enum method_e method;
    unsigned nrounds;
    union{
#define RNGNxW_TPL(base, N, W) base##N##x##W##_kat base##N##x##W##_data;
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL
	// Sigh... For those platforms that lack uint64_t, carve
	//   out 128 bytes for the counter, key, expected, and computed.
	char justbytes[128]; 
    }u;
};

#define RNGNxW_TPL(base, N, W) \
int read_##base##N##x##W(const char *line, kat_instance* tinst){        \
    size_t i;                                                           \
    int nchar;                                                          \
    const char *p = line;                                               \
    char *newp;                                                         \
    size_t nkey = sizeof(tinst->u.base##N##x##W##_data.ukey.v)/sizeof(tinst->u.base##N##x##W##_data.ukey.v[0]); \
    tinst->method = base##N##x##W##_e;                                  \
    sscanf(p,  "%u%n", &tinst->nrounds, &nchar);                        \
    p += nchar;                                                         \
    for(i=0;  i<N; ++i){                                                \
        tinst->u.base##N##x##W##_data.ctr.v[i] = strtou##W(p, &newp, 16); \
        p = newp;                                                       \
    }                                                                   \
    for(i=0; i<nkey; ++i){                                              \
        tinst->u.base##N##x##W##_data.ukey.v[i] = strtou##W(p, &newp, 16); \
        p = newp;                                                       \
    }                                                                   \
    for(i=0;  i<N; ++i){                                                \
        tinst->u.base##N##x##W##_data.expected.v[i] = strtou##W(p, &newp, 16); \
        p = newp;                                                       \
    }                                                                   \
    /* set the computed to 0xca.  If the test fails to set computed, we'll see cacacaca in the FAILURE notices */ \
    memset(tinst->u.base##N##x##W##_data.computed.v, 0xca, sizeof(tinst->u.base##N##x##W##_data.computed.v));                  \
    return 1;                                                           \
}
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

#define RNGNxW_TPL(base, N, W) \
void report_##base##N##x##W##error(int &nfailed, const kat_instance *ti){ \
 size_t i;                                                     \
 size_t nkey = sizeof(ti->u.base##N##x##W##_data.ukey.v)/sizeof(ti->u.base##N##x##W##_data.ukey.v[0]); \
 std::stringstream ss;                                         \
 ss << "FAIL:  expected: ";                                    \
 ss << #base #N "x" #W " " << ti->nrounds;                     \
 for(i=0; i<N; ++i){                                           \
     ss << " "; prtu##W(ss, ti->u.base##N##x##W##_data.ctr.v[i]); \
 }                                                             \
 for(i=0; i<nkey; ++i){                                        \
     ss << " "; prtu##W(ss, ti->u.base##N##x##W##_data.ukey.v[i]); \
 }                                                             \
 for(i=0; i<N; ++i){                                           \
     ss << " "; prtu##W(ss, ti->u.base##N##x##W##_data.expected.v[i]); \
 }                                                             \
 ss << "\n";                                                   \
                                                               \
 ss << "FAIL:  computed: ";                                    \
 ss << #base #N "x" #W " " << ti->nrounds;                     \
 for(i=0; i<N; ++i){                                           \
     ss << " "; prtu##W(ss, ti->u.base##N##x##W##_data.ctr.v[i]); \
 }                                                             \
 for(i=0; i<nkey; ++i){                                        \
     ss << " "; prtu##W(ss, ti->u.base##N##x##W##_data.ukey.v[i]); \
 }                                                             \
 for(i=0; i<N; ++i){                                           \
     ss << " "; prtu##W(ss, ti->u.base##N##x##W##_data.computed.v[i]); \
 }                                                             \
 ss << "\n";                                                   \
 FAIL() << ss.str();                                           \
 nfailed++;                                                    \
}
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

struct UnknownKatTracker {
    const static int MAXUNKNOWNS = 20;
    int num_unknowns = 0;
    const char *unknown_names[MAXUNKNOWNS];
    int unknown_counts[MAXUNKNOWNS];
};

void register_unknown(UnknownKatTracker &ukt, const char *name){
    int i;
    for(i=0; i< ukt.num_unknowns; ++i){
        if( strcmp(name, ukt.unknown_names[i]) == 0 ){
            ukt.unknown_counts[i]++;
            return;
        }
    }
    if( i >= ukt.MAXUNKNOWNS ){
        FAIL() << "Too many unknown rng types. Bye.\n";
    }
    ukt.num_unknowns++;
    ukt.unknown_names[i] = ntcsdup(name);
    ukt.unknown_counts[i] = 1;
}

void base_rng_test_arrange(const char *line, kat_instance* tinst, UnknownKatTracker &ukt, bool &flag){
    int nchar;
    char name[LINESIZE];
    if( line[0] == '#') {
        flag = false;
        return;
    }                                       
    sscanf(line, "%s%n", name, &nchar);
    /* skip any tests that require AESNI */ 
    if(strncmp(name, "aes", 3)==0 || strncmp(name, "ars", 3)==0){
        register_unknown(ukt, name);
        flag = false;
        return;
    }
#define RNGNxW_TPL(base, N, W) \
    if(strcmp(name, #base #N "x" #W) == 0)  {                    \
        flag = (bool) read_##base##N##x##W(line+nchar, tinst);   \
        return;                                                  \
    }                                                            
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

    register_unknown(ukt, name);
    flag = false;
    return;
}

static int murng_reported;
static int engine_reported;

template <typename GEN>
void base_rng_test_act(kat_instance* ti){
    GEN g;
    struct gdata{
        typename GEN::ctr_type ctr;
        typename GEN::ukey_type ukey;
        typename GEN::ctr_type expected;
        typename GEN::ctr_type computed;
    };
    gdata data;
    // use memcpy.  A reinterpret_cast would violate strict aliasing.
    std::memcpy(&data, &ti->u, sizeof(data));
    data.computed = g(data.ctr, data.ukey);

    // Before we return, let's make sure that MicroURNG<GEN,1> and
    // Engine<GEN> work as expeccted.  This doesn't really "fit" the
    // execution model of kat.c, which just expects us to fill in
    // ti->u.computed, so we report the error by failing to write back
    // the computed data item in the (hopefully unlikely) event that
    // things don't match up as expected.
    int errs = 0;

    // MicroURNG:  throws if the top 32 bits of the high word of ctr
    // are non-zero.
    typedef typename GEN::ctr_type::value_type value_type;
    
    value_type hibits = data.ctr[data.ctr.size()-1]>>( std::numeric_limits<value_type>::digits - 32 );
    try{
        r123::MicroURNG<GEN> urng(data.ctr, data.ukey);
        if(hibits)
            errs++; // Should have thrown.
        for (size_t i = 0; i < data.expected.size(); i++) {
	    size_t j = data.expected.size() - i - 1;
	    if (data.expected[j] != urng()) {
                errs++;
	    }
	}
    }catch(std::runtime_error& /*ignored*/){
        // A runtime_error is expected from the constructor
        // when hibit is set.
        if(!hibits)
            errs++;
    }
    if(errs && (murng_reported++ == 0))
        std::cerr << "Error in MicroURNG<GEN>, will appear as \"computed\" value of zero in error summary\n";

    // Engine
    // N.B.  exercising discard() arguably belongs in ut_Engine.cpp
    typedef r123::Engine<GEN> Etype;
    typedef typename GEN::ctr_type::value_type value_type;
    Etype e(data.ukey);
    typename GEN::ctr_type c = data.ctr;
    value_type c0;
    if( c[0] > 0 ){
        c0 = c[0]-1;
    }else{
        // N.B.  Assume that if c[0] is 0, then so are all the
        // others.  Arrange to "roll over" to {0,..,0} on the first
        // counter-increment.  Alternatively, we could just
        // skip the test for this case...
        c.fill(std::numeric_limits<value_type>::max());
        c0 = c[0];
    }
    c[0] /= 3;
    e.setcounter(c, 0);
    if( c0 > c[0] ){
        // skip one value by calling  e()
        (void)e();
        if (c0 > c[0]+1) {
	    // skip many values by calling discard()
	    R123_ULONG_LONG ndiscard = (c0 - c[0] - 1);
            // Take care not to overflow the long long
            if( ndiscard >= std::numeric_limits<R123_ULONG_LONG>::max() / c.size() ){
                for(size_t j=0; j<c.size(); ++j){
                    e.discard(ndiscard);
                }
            }else{
                ndiscard *= c.size();
                e.discard(ndiscard);
            }
	}
	// skip a few more by calling e().
	for (size_t i = 1; i < c.size(); i++) {
	    (void) e();
	}
        // we should be back to where we started...
    }
    for (size_t i = 0; i < data.expected.size(); i++) {
	value_type val = e();
	size_t j = data.expected.size() - i - 1;
	if (data.expected[j] != val) {
            std::cerr << std::hex;
            std::cerr << "Engine check, j=" << j << " expected: " << data.expected[j] << " val: " << val << "\n";
	    errs++;
            if(engine_reported++ == 0)
                std::cerr << "Error in Engine<GEN, 1>, will appear as \"computed\" value of zero in error summary\n";
	}
    }

    // Signal an error to the caller by *not* copying back
    // the computed data object into the ti
    if(errs == 0)
        std::memcpy(&ti->u, &data, sizeof(data));
}

void base_rng_test_assert(int &nfailed, const kat_instance *tests, unsigned ntests){
    unsigned i;
    char zeros[512] = {0};
    for(i=0; i<ntests; ++i){
        const kat_instance *ti = &tests[i];
        switch(tests[i].method){
#define RNGNxW_TPL(base, N, W) case base##N##x##W##_e: \
            if (memcmp(zeros, ti->u.base##N##x##W##_data.expected.v, N*W/8)==0){ \
                FAIL() << "kat expected all zeros?   Something is wrong with the test harness!\n"; \
                nfailed++; \
            } \
            if (memcmp(ti->u.base##N##x##W##_data.computed.v, ti->u.base##N##x##W##_data.expected.v, N*W/8)) \
		report_##base##N##x##W##error(nfailed, ti); \
	    break;
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL
        case last: ;
        }
    }
}

void run_all_base_rng_kats() {
    kat_instance *tests;
    unsigned t, ntests = 1000;
    char linebuf[LINESIZE];
    FILE *inpfile;
    const char *p;
    const char *inname;
    int nfailed = 0;

    UnknownKatTracker ukt{};

    inname = "./r123_kat_vectors.txt";
	inpfile = fopen(inname, "r");
	if (inpfile == NULL)
        FAIL() << "Error opening input file " << inname << " for reading. Received error code " << errno << "\n";

    if ((p = getenv("KATC_VERBOSE")) != NULL)
	    verbose = atoi(p);
  
    if ((p = getenv("KATC_DEBUG")) != NULL)
	    debug = atoi(p);

    tests = (kat_instance *) malloc(sizeof(tests[0])*ntests);
    if (tests == NULL) {
        FAIL() << "Could not allocate " << (unsigned long) ntests << " bytes for tests\n";
    }
    t = 0;
    while (fgets(linebuf, sizeof linebuf, inpfile) != NULL) {
        if( t == ntests ) {
            ntests *= 2;
            tests = (kat_instance *)realloc(tests, sizeof(tests[0])*ntests);
            if (tests == NULL) {
                FAIL() << "Could not grow tests to " << (unsigned long) ntests << " bytes.\n";
            }
        }
        bool flag = false;
        base_rng_test_arrange(linebuf, &tests[t], ukt, flag);
        if( flag )
            ++t;
    }
    if(t==ntests){
	    FAIL() << "No more space for tests?  Recompile with a larger ntests\n";
    }
    tests[t].method = last;

    for(int i=0; i< ukt.num_unknowns; ++i){
        printf("%d test vectors of type %s skipped\n", ukt.unknown_counts[i], ukt.unknown_names[i]);
    }
    printf("Perform %lu tests.\n", (unsigned long)t);
    using std::map;
    using std::pair;
    using std::make_pair;
    typedef map<pair<method_e, unsigned>, void (*)(kat_instance *)> genmap_t;
    genmap_t genmap;
    // In C++1x, this could be staticly declared with an initializer list.
    genmap[make_pair(threefry2x32_e, 13u)] = base_rng_test_act<r123::Threefry2x32_R<13> >;
    genmap[make_pair(threefry2x32_e, 20u)] = base_rng_test_act<r123::Threefry2x32_R<20> >;
    genmap[make_pair(threefry2x32_e, 32u)] = base_rng_test_act<r123::Threefry2x32_R<32> >;

    genmap[make_pair(threefry4x32_e, 13u)] = base_rng_test_act<r123::Threefry4x32_R<13> >;
    genmap[make_pair(threefry4x32_e, 20u)] = base_rng_test_act<r123::Threefry4x32_R<20> >;
    genmap[make_pair(threefry4x32_e, 72u)] = base_rng_test_act<r123::Threefry4x32_R<72> >;

    #if R123_USE_64BIT
    genmap[make_pair(threefry2x64_e, 13u)] = base_rng_test_act<r123::Threefry2x64_R<13> >;
    genmap[make_pair(threefry2x64_e, 20u)] = base_rng_test_act<r123::Threefry2x64_R<20> >;
    genmap[make_pair(threefry2x64_e, 32u)] = base_rng_test_act<r123::Threefry2x64_R<32> >;

    genmap[make_pair(threefry4x64_e, 13u)] = base_rng_test_act<r123::Threefry4x64_R<13> >;
    genmap[make_pair(threefry4x64_e, 20u)] = base_rng_test_act<r123::Threefry4x64_R<20> >;
    genmap[make_pair(threefry4x64_e, 72u)] = base_rng_test_act<r123::Threefry4x64_R<72> >;
    #endif

    genmap[make_pair(philox2x32_e, 7u)] = base_rng_test_act<r123::Philox2x32_R<7> >;
    genmap[make_pair(philox2x32_e, 10u)] = base_rng_test_act<r123::Philox2x32_R<10> >;
    genmap[make_pair(philox4x32_e, 7u)] = base_rng_test_act<r123::Philox4x32_R<7> >;
    genmap[make_pair(philox4x32_e, 10u)] = base_rng_test_act<r123::Philox4x32_R<10> >;

    #if R123_USE_PHILOX_64BIT
    genmap[make_pair(philox2x64_e, 7u)] = base_rng_test_act<r123::Philox2x64_R<7> >;
    genmap[make_pair(philox2x64_e, 10u)] = base_rng_test_act<r123::Philox2x64_R<10> >;
    genmap[make_pair(philox4x64_e, 7u)] = base_rng_test_act<r123::Philox4x64_R<7> >;
    genmap[make_pair(philox4x64_e, 10u)] = base_rng_test_act<r123::Philox4x64_R<10> >;
    #endif

    unsigned i;
    for(i=0; i<t; ++i){
        kat_instance *ti = &tests[i];
        genmap_t::iterator p = genmap.find(make_pair(ti->method, ti->nrounds));
        if(p == genmap.end())
            throw std::runtime_error("pair<generator, nrounds> not in map.  You probably need to add more genmap entries.");

        p->second(ti);
        // ^ That prepares the test data
    }

    base_rng_test_assert(nfailed, tests, t);
    free(tests);
    if(nfailed != 0)
        FAIL() << "Failed " << nfailed << " out of " << t << std::endl;
    return;
}

// MARK: histogram test

using namespace r123;

template <typename T>
typename r123::make_unsigned<T>::type U(T x){ return x; }

template <typename T>
typename r123::make_signed<T>::type S(T x){ return x; }
        
#define Chk(u, Rng, Ftype, _nfail_, _refhist_) do{                            \
        chk<Ftype, Rng>(#u, #Rng, #Ftype, &u<Ftype, Rng::ctr_type::value_type>, _nfail_, _refhist_); \
    }while(0)

template<typename Ftype, typename RNG, typename Utype>
void chk(const std::string& fname, const std::string& rngname, const std::string& ftypename, Utype f, int &nfail, std::map<std::string, std::string> &refmap){
    std::string key = fname + " " + rngname + " " + ftypename;
    RNG rng;
    typedef typename RNG::ukey_type ukey_type;
    typedef typename RNG::ctr_type ctr_type;
    typedef typename RNG::key_type key_type;

    ctr_type c = {{}};
    ukey_type uk = {{}};
    key_type k = uk;
    // 26 bins - 13 greater than 0 and 13 less.  Why 13?  Because a
    // prime number seems less likely to tickle the rounding-related
    // corner cases, which is aruably both good and bad.
    const int NBINS=26;
    
    int hist[NBINS] = {};
    for(int i=0; i<1000; ++i){
        c = c.incr();
        ctr_type r = rng(c, k);
        for(int j=0; j<ctr_type::static_size; ++j){
            Ftype u = f(r[j]);
            //printf("%s %llx, %.17g\n", key.c_str(), (long long)r[j], (double)u);
            R123_ASSERT( u >= -1.);
            R123_ASSERT( u <= 1.);
            int idx = (int) ((u + Ftype(1.))*Ftype(NBINS/2));
            hist[idx]++;
        }
    }
    std::ostringstream oss;
    for(int i=0; i<NBINS; ++i){
        oss << " " << hist[i];
    }
    if( oss.str() != refmap[key] ){
        std::stringstream ss{};
        ss << "MISMTACH : " << key << ":\n\tcomputed histogram  = " << oss.str() << "\n\treference histogram = " << refmap[key] << std::endl;
        nfail++;
        FAIL() << ss.str();
    }
}

std::map<std::string, std::string> get_ut_uniform_refmap(){
    using std::string;
    std::map<string, string> refmap{};
    refmap[string("u01 Threefry4x32 float")] = string(" 0 0 0 0 0 0 0 0 0 0 0 0 0 301 330 326 320 295 291 298 287 305 307 310 316 314");
    refmap[string("u01 Threefry4x32 double")]= string(" 0 0 0 0 0 0 0 0 0 0 0 0 0 301 330 326 320 295 291 298 287 305 307 310 316 314");
    refmap[string("u01 Threefry4x32 long double")] = string(" 0 0 0 0 0 0 0 0 0 0 0 0 0 301 330 326 320 295 291 298 287 305 307 310 316 314");
    refmap[string("u01 Threefry4x64 float")] = string(" 0 0 0 0 0 0 0 0 0 0 0 0 0 308 295 322 300 316 291 311 289 346 297 310 340 275");
    refmap[string("u01 Threefry4x64 double")] = string(" 0 0 0 0 0 0 0 0 0 0 0 0 0 308 295 322 300 316 291 311 289 346 297 310 340 275");
    refmap[string("u01 Threefry4x64 long double")] = string(" 0 0 0 0 0 0 0 0 0 0 0 0 0 308 295 322 300 316 291 311 289 346 297 310 340 275");
    refmap[string("uneg11 Threefry4x32 float")] = string(" 156 139 148 146 159 148 159 168 142 160 156 161 153 143 158 150 180 174 152 163 157 129 166 151 140 142");
    refmap[string("uneg11 Threefry4x32 double")] = string(" 156 139 148 146 159 148 159 168 142 160 156 161 153 143 158 150 180 174 152 163 157 129 166 151 140 142");
    refmap[string("uneg11 Threefry4x32 long double")] = string( " 156 139 148 146 159 148 159 168 142 160 156 161 153 143 158 150 180 174 152 163 157 129 166 151 140 142");
    refmap[string("uneg11 Threefry4x64 float")] = string( " 159 141 148 184 162 142 155 137 173 187 153 140 135 164 144 146 149 151 171 152 148 137 179 146 145 152");
    refmap[string("uneg11 Threefry4x64 double")] = string( " 159 141 148 184 162 142 155 137 173 187 153 140 135 164 144 146 149 151 171 152 148 137 179 146 145 152");
    refmap[string("uneg11 Threefry4x64 long double")] = string( " 159 141 148 184 162 142 155 137 173 187 153 140 135 164 144 146 149 151 171 152 148 137 179 146 145 152");
    refmap[string("u01fixedpt Threefry4x32 float")] = string( " 0 0 0 0 0 0 0 0 0 0 0 0 0 301 330 326 320 295 291 298 287 305 307 310 316 314");
    refmap[string("u01fixedpt Threefry4x32 double")] = string( " 0 0 0 0 0 0 0 0 0 0 0 0 0 301 330 326 320 295 291 298 287 305 307 310 316 314");
    refmap[string("u01fixedpt Threefry4x32 long double")] = string( " 0 0 0 0 0 0 0 0 0 0 0 0 0 301 330 326 320 295 291 298 287 305 307 310 316 314");
    refmap[string("u01fixedpt Threefry4x64 float")] = string( " 0 0 0 0 0 0 0 0 0 0 0 0 0 308 295 322 300 316 291 311 289 346 297 310 340 275");
    refmap[string("u01fixedpt Threefry4x64 double")] = string( " 0 0 0 0 0 0 0 0 0 0 0 0 0 308 295 322 300 316 291 311 289 346 297 310 340 275");
    refmap[string("u01fixedpt Threefry4x64 long double")] = string( " 0 0 0 0 0 0 0 0 0 0 0 0 0 308 295 322 300 316 291 311 289 346 297 310 340 275");
    return refmap;
}

void run_ut_uniform(){
    auto refmap = get_ut_uniform_refmap();
    int nfail = 0;
    // 18 tests:  3 functions (u01, uneg11, u01fixedpt)
    //          x 2 input sizes (32 bit or 64 bit)
    //          x 3 output sizes (float, double, long double)
    Chk(u01, Threefry4x32, float, nfail, refmap);
    Chk(u01, Threefry4x32, double, nfail, refmap);
    Chk(u01, Threefry4x32, long double, nfail, refmap);

#if R123_USE_64BIT
    Chk(u01, Threefry4x64, float, nfail, refmap);
    Chk(u01, Threefry4x64, double, nfail, refmap);
    Chk(u01, Threefry4x64, long double, nfail, refmap);
#endif

    Chk(uneg11, Threefry4x32, float, nfail, refmap);
    Chk(uneg11, Threefry4x32, double, nfail, refmap);
    Chk(uneg11, Threefry4x32, long double, nfail, refmap);

#if R123_USE_64BIT
    Chk(uneg11, Threefry4x64, float, nfail, refmap);
    Chk(uneg11, Threefry4x64, double, nfail, refmap);
    Chk(uneg11, Threefry4x64, long double, nfail, refmap);
#endif
    
    Chk(u01fixedpt, Threefry4x32, float, nfail, refmap);
    Chk(u01fixedpt, Threefry4x32, double, nfail, refmap);
    Chk(u01fixedpt, Threefry4x32, long double, nfail, refmap);

#if R123_USE_64BIT
    Chk(u01fixedpt, Threefry4x64, float, nfail, refmap);
    Chk(u01fixedpt, Threefry4x64, double, nfail, refmap);
    Chk(u01fixedpt, Threefry4x64, long double, nfail, refmap);
#endif

    ASSERT_EQ(nfail, 0);
    return;
}

// MARK: my tests + Googletest

class TestRNGState : public ::testing::Test {

    protected:

    void test_uint_key_constructors() {
        using RNG = r123::Philox4x32;
        int len_k = RNG::key_type::static_size;
        ASSERT_EQ(len_k, 2);
        // No-arugment constructor
        RandBLAS::RNGState<RNG> s;
        ASSERT_EQ(s.key[0], 0);
        ASSERT_EQ(s.key[1], 0);
        for (int i = 0; i < 4; ++i) {
            ASSERT_EQ(s.counter[i], 0) << "Failed at index " << i;
        }
        // unsigned-int constructor
        RandBLAS::RNGState<RNG> t(42);
        ASSERT_EQ(t.key[0], 42);
        ASSERT_EQ(t.key[1], 0);
        for (int i = 0; i < 4; ++i) {
            ASSERT_EQ(t.counter[i], 0) << "Failed at index " << i;
        }
        return;
    }
};

TEST_F(TestRNGState, uint_key_constructors) {
    test_uint_key_constructors();
}


class TestRandom123 : public ::testing::Test { 

    protected:
    
    static void test_incr() {
        using RNG = r123::Philox4x32;
        RandBLAS::RNGState<RNG> s(0);
        // The "counter" array of s is a 4*32=128 bit unsigned integer.
        //
        //      Each block is interpreted in the usual way (i.e., no need to consider differences
        //      between big-endian and little-endian representations). 
        //
        //      Looking across blocks, we read as as a little-endian number in base IMAX = 2^32 - 1.
        //      That is, if we initialize s.counter = {0,0,0,0} and then call s.counter.incr(IMAX),
        //      we should have s.counter = {IMAX, 0, 0, 0}, and if we make another call 
        //      s.counter.incr(9), then we should see s.counter = {8, 1, 0, 0}. Put another way,
        //      if c = s.counter, then we have
        //
        //        (128-bit integer) c == c[0] + 2^{32}*c[1] +  2^{64}*c[2] + 2^{96}*c[3]  (mod 2^128 - 1)
        //
        //       where 0 <= c[i] <= IMAX
        //
        uint64_t i32max = std::numeric_limits<uint32_t>::max();
        auto c = s.counter;
        ASSERT_EQ(c[0], 0);
        ASSERT_EQ(c[1], 0);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);

        c.incr(i32max);
        ASSERT_EQ(c[0], i32max);
        ASSERT_EQ(c[1], 0);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);

        c.incr(1);
        ASSERT_EQ(c[0], 0);
        ASSERT_EQ(c[1], 1);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);

        c.incr(3);
        ASSERT_EQ(c[0], 3);
        ASSERT_EQ(c[1], 1);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);

        uint64_t two32  = ((uint64_t) 1) << 32;

        c = {0,0,0,0};
        c.incr(two32-1);
        ASSERT_EQ(c[0], i32max);
        ASSERT_EQ(c[1], 0);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);

        c = {0,0,0,0};
        c.incr(two32);
        ASSERT_EQ(c[0], 0);
        ASSERT_EQ(c[1], 1);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);

        // Let's construct 2^32 * (2^32 - 1), which is equal to (ctr_type) {0, (uint32_t) i32max, 0, 0}.
        //
        //  Do this using the identity 
        //      2^32 * (2^32 - 1) == 2^64 - 2^32
        //                        == 2^63 + 2^63 - 2^32.
        //
        // Then construct 2^64, using 2^64 = (2^63) + (2^63 - 2^32) + (2^32)
        uint64_t two63  = ((uint64_t) 1) << 63;
        c = {0,0,0,0};
        c.incr(two63);
        c.incr(two63 - two32);
        ASSERT_EQ(c[0], 0);
        ASSERT_EQ(c[1], i32max);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 0);
        c.incr(two32);
        ASSERT_EQ(c[0], 0);
        ASSERT_EQ(c[1], 0);
        ASSERT_EQ(c[2], 1);
        ASSERT_EQ(c[3], 0);

        c = {(uint32_t) i32max, (uint32_t) i32max, (uint32_t) i32max, 0};
        c.incr(1);
        ASSERT_EQ(c[0], 0);
        ASSERT_EQ(c[1], 0);
        ASSERT_EQ(c[2], 0);
        ASSERT_EQ(c[3], 1);
        return;
    }
};

TEST_F(TestRandom123, base_generators) {
    run_all_base_rng_kats();
}

TEST_F(TestRandom123, uniform_histograms) {
    run_ut_uniform();
}

TEST_F(TestRandom123, big_incr) {
    test_incr();
}