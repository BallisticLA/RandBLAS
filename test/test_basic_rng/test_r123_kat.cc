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
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/ars.h>
#include <Random123/aes.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <math.h>
#include <Random123/features/compilerfeatures.h>
#include <sstream>

#include <iostream>
#include <iomanip>
#include <cassert>
#include <charconv>
#include <cassert>
#include <system_error>
#include <cstdint>
#include <Random123/uniform.hpp>
#include <map>
#include <string>


#define LINESIZE 1024

#if R123_USE_AES_NI
    int have_aesni = haveAESNI();
#else
    int have_aesni = 0;
#endif
int verbose = 0;
int debug = 0;

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
#ifdef _MSC_FULL_VER
#define strtoull _strtoui64
#endif
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

// Specializations for uint32_t and uint64_t
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

enum method_e{
#define RNGNxW_TPL(base, N, W) base##N##x##W##_e,
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL
    last
};

#define RNGNxW_TPL(base, N, W)                       \
    typedef struct {                                 \
        base##N##x##W##_ctr_t ctr;                   \
        base##N##x##W##_ukey_t ukey;                 \
        base##N##x##W##_ctr_t expected;              \
        base##N##x##W##_ctr_t computed;              \
    } base##N##x##W##_kat;
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

typedef struct{
    enum method_e method;
    unsigned nrounds;
    union{
#define RNGNxW_TPL(base, N, W) base##N##x##W##_kat base##N##x##W##_data;
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL
	/* Sigh... For those platforms that lack uint64_t, carve
	   out 128 bytes for the counter, key, expected, and computed. */
	char justbytes[128]; 
    }u;
} kat_instance;

void host_execute_tests(kat_instance *tests, unsigned ntests);
                
/* Keep track of the test vectors that we don't know how to deal with: */
#define MAXUNKNOWNS 20

struct UnknownKatTracker {
    int num_unknowns;
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
    if( i >= MAXUNKNOWNS ){
        FAIL() << "Too many unknown rng types. Bye.\n";
    }
    ukt.num_unknowns++;
    ukt.unknown_names[i] = ntcsdup(name);
    ukt.unknown_counts[i] = 1;
}

/* read_<GEN>NxW */
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


/* readtest:  dispatch to one of the read_<GEN>NxW functions */
static int readtest(UnknownKatTracker &ukt, const char *line, kat_instance* tinst){
    int nchar;
    char name[LINESIZE];
    if( line[0] == '#') return 0;                                       
    sscanf(line, "%s%n", name, &nchar);
    if(!have_aesni){
        /* skip any tests that require AESNI */ 
        if(strncmp(name, "aes", 3)==0 ||
           strncmp(name, "ars", 3)==0){
            register_unknown(ukt, name);
            return 0;
        }
    }
#define RNGNxW_TPL(base, N, W) if(strcmp(name, #base #N "x" #W) == 0) return read_##base##N##x##W(line+nchar, tinst);
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

    register_unknown(ukt, name);
    return 0;
}

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

// dispatch to one of the report_<GEN>NxW() functions
void analyze_tests(int &nfailed, const kat_instance *tests, unsigned ntests){
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

#define NTESTS 1000

void run_base_rng_kat() {
    kat_instance *tests;
    unsigned t, ntests = NTESTS;
    char linebuf[LINESIZE];
    FILE *inpfile;
    const char *p;
    const char *inname;
    int nfailed;

    UnknownKatTracker ukt{};

    inname = "./kat_vectors.txt";
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
        if( readtest(ukt, linebuf, &tests[t]) )
            ++t;
    }
    if(t==ntests){
	    FAIL() << "No more space for tests?  Recompile with a larger NTESTS\n";
    }
    tests[t].method = last; // N.B  *not* t++ - the 'ntests' value passed to host_execute_tests does not count the 'last' one.

    for(int i=0; i< ukt.num_unknowns; ++i){
        printf("%d test vectors of type %s skipped\n", ukt.unknown_counts[i], ukt.unknown_names[i]);
    }
    printf("Perform %lu tests.\n", (unsigned long)t);
    host_execute_tests(tests, t);

    analyze_tests(nfailed, tests, t);
    free(tests);
    if(nfailed != 0)
        FAIL() << "Failed " << nfailed << " out of " << t << std::endl;
    return;
}



// With C++, it's a little trickier to create the mapping from
// method-name/round-count to functions
// because the round-counts are template arguments that have to be
// specified at compile-time.  Thus, we can't just do #define RNGNxW_TPL
// and #include "r123_rngNxW.mm".  We have to build a static map from:
//  pair<generator, rounds> to functions that apply the right generator
// with the right number of rounds.

#ifdef _MSC_FULL_VER
// Engines have multiple copy constructors, quite legal C++, disable MSVC complaint
#pragma warning (disable : 4521)
#endif

#include <map>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <Random123/MicroURNG.hpp>
#include <Random123/conventional/Engine.hpp>

using namespace std;

typedef map<pair<method_e, unsigned>, void (*)(kat_instance *)> genmap_t;
genmap_t genmap;

void dev_execute_tests(kat_instance *tests, unsigned ntests){
    unsigned i;
    for(i=0; i<ntests; ++i){
        kat_instance *ti = &tests[i];
        genmap_t::iterator p = genmap.find(make_pair(ti->method, ti->nrounds));
        if(p == genmap.end())
            throw std::runtime_error("pair<generator, nrounds> not in map.  You probably need to add more genmap entries in kat_cpp.cpp");

        p->second(ti);
        // TODO: check that the corresponding Engine and MicroURNG
        //  return the same values.  Note that we have ut_Engine and
        //  ut_MicroURNG, which check basic functionality, but they
        //  don't have the breadth of the kat_vectors.
    }
}

static int murng_reported;
static int engine_reported;

template <typename GEN>
void do_test(kat_instance* ti){
    GEN g;
    struct gdata{
        typename GEN::ctr_type ctr;
        typename GEN::ukey_type ukey;
        typename GEN::ctr_type expected;
        typename GEN::ctr_type computed;
    };
    gdata data;
    // use memcpy.  A reinterpret_cast would violate strict aliasing.
    memcpy(&data, &ti->u, sizeof(data));
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
        cerr << "Error in MicroURNG<GEN>, will appear as \"computed\" value of zero in error summary\n";

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
            cerr << hex;
            cerr << "Engine check, j=" << j << " expected: " << data.expected[j] << " val: " << val << "\n";
	    errs++;
            if(engine_reported++ == 0)
                cerr << "Error in Engine<GEN, 1>, will appear as \"computed\" value of zero in error summary\n";
	}
    }

    // Signal an error to the caller by *not* copying back
    // the computed data object into the ti
    if(errs == 0)
        memcpy(&ti->u, &data, sizeof(data));
}

void host_execute_tests(kat_instance *tests, unsigned ntests){
    // In C++1x, this could be staticly declared with an initializer list.
    genmap[make_pair(threefry2x32_e, 13u)] = do_test<r123::Threefry2x32_R<13> >;
    genmap[make_pair(threefry2x32_e, 20u)] = do_test<r123::Threefry2x32_R<20> >;
    genmap[make_pair(threefry2x32_e, 32u)] = do_test<r123::Threefry2x32_R<32> >;
#if R123_USE_64BIT
    genmap[make_pair(threefry2x64_e, 13u)] = do_test<r123::Threefry2x64_R<13> >;
    genmap[make_pair(threefry2x64_e, 20u)] = do_test<r123::Threefry2x64_R<20> >;
    genmap[make_pair(threefry2x64_e, 32u)] = do_test<r123::Threefry2x64_R<32> >;
#endif

    genmap[make_pair(threefry4x32_e, 13u)] = do_test<r123::Threefry4x32_R<13> >;
    genmap[make_pair(threefry4x32_e, 20u)] = do_test<r123::Threefry4x32_R<20> >;
    genmap[make_pair(threefry4x32_e, 72u)] = do_test<r123::Threefry4x32_R<72> >;
#if R123_USE_64BIT
    genmap[make_pair(threefry4x64_e, 13u)] = do_test<r123::Threefry4x64_R<13> >;
    genmap[make_pair(threefry4x64_e, 20u)] = do_test<r123::Threefry4x64_R<20> >;
    genmap[make_pair(threefry4x64_e, 72u)] = do_test<r123::Threefry4x64_R<72> >;
#endif

    genmap[make_pair(philox2x32_e, 7u)] = do_test<r123::Philox2x32_R<7> >;
    genmap[make_pair(philox2x32_e, 10u)] = do_test<r123::Philox2x32_R<10> >;
    genmap[make_pair(philox4x32_e, 7u)] = do_test<r123::Philox4x32_R<7> >;
    genmap[make_pair(philox4x32_e, 10u)] = do_test<r123::Philox4x32_R<10> >;

#if R123_USE_PHILOX_64BIT
    genmap[make_pair(philox2x64_e, 7u)] = do_test<r123::Philox2x64_R<7> >;
    genmap[make_pair(philox2x64_e, 10u)] = do_test<r123::Philox2x64_R<10> >;
    genmap[make_pair(philox4x64_e, 7u)] = do_test<r123::Philox4x64_R<7> >;
    genmap[make_pair(philox4x64_e, 10u)] = do_test<r123::Philox4x64_R<10> >;
#endif

#if R123_USE_AES_NI
    genmap[make_pair(aesni4x32_e, 10u)] = do_test<r123::AESNI4x32 >;
    genmap[make_pair(ars4x32_e, 7u)] = do_test<r123::ARS4x32_R<7> >;
    genmap[make_pair(ars4x32_e, 10u)] = do_test<r123::ARS4x32_R<10> >;
#endif

    dev_execute_tests(tests, ntests);
}


using namespace r123;

template <typename T>
typename r123::make_unsigned<T>::type U(T x){ return x; }

template <typename T>
typename r123::make_signed<T>::type S(T x){ return x; }
        
#define Chk(u, Rng, Ftype, _nfail_, _refhist_) do{                            \
        chk<Ftype, Rng>(#u, #Rng, #Ftype, &u<Ftype, Rng::ctr_type::value_type>, _nfail_, _refhist_); \
    }while(0)

void RefHist(std::map<std::string, std::string> &refmap, const char* k, const char *v){
    refmap[std::string(k)] = std::string(v);
}

void fillrefhist(std::map<std::string, std::string> &refmap){
    #include "ut_uniform_reference.mm"
}

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

void run_ut_uniform(){
    std::map<std::string, std::string> refmap{};
    fillrefhist(refmap);
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

class TestRandom123KnownAnswers : public ::testing::Test { };

TEST_F(TestRandom123KnownAnswers, base_generators) {
    run_base_rng_kat();
}

TEST_F(TestRandom123KnownAnswers, uniform_histograms) {
    run_ut_uniform();
}
