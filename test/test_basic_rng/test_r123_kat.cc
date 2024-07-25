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

extern char *progname;
extern int debug;
extern int verbose;

/* strdup may or may not be in string.h, depending on the value
   of the pp-symbol _XOPEN_SOURCE and other arcana.  Just
   do it ourselves.
   Mnemonic:  "ntcs" = "nul-terminated character string" */
char *ntcsdup(const char *s){
    char *p = (char *)malloc(strlen(s)+1);
    strcpy(p, s);
    return p;
}

/* MSVC doesn't know about strtoull.  Strictly speaking, strtoull
   isn't standardized in C++98, either, but that seems not to be a
   problem so we blissfully ignore it and use strtoull (or its MSVC
   equivalent, _strtoui64) in both C and C++.  If strtoull in C++
   becomes a problem, we can adopt the prtu strategy (see below) and
   write C++ versions of strtouNN, that use an istringstream
   instead. */
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

/* Strict C++98 doesn't grok %llx or unsigned long long, and with
   aggressive error-checking, e.g., g++ -pedantic -Wall, will refuse
   to compile code like:

     fprintf(stderr, "%llx", (R123_ULONG_LONG)v);

   On the other hand, when compiling to a 32-bit target, the only
   64-bit type is long long, so we're out of luck if we can't use llx.
   A portable, almost-standard way to do I/O on uint64_t values in C++
   is to use bona fide C++ I/O streams.  We are still playing
   fast-and-loose with standards because C++98 doesn't have <stdint.h>
   and hence doesn't even guarantee that there's a uint64_t, much less
   that the insertion operator<<(ostream&) works correctly with
   whatever we've typedef'ed to uint64_t in
   <features/compilerfeatures.h>.  Hope for the best... */
#include <iostream>
#include <limits>
template <typename T>
void prtu(T val){
    using namespace std;
    cerr.width(std::numeric_limits<T>::digits/4);
    char prevfill = cerr.fill('0');
    ios_base::fmtflags prevflags = cerr.setf(ios_base::hex, ios_base::basefield);
    cerr << val;
    cerr.flags(prevflags);
    cerr.fill(prevfill);
    assert(!cerr.bad());
}
void prtu32(uint32_t v){ prtu(v); }
void prtu64(uint64_t v){ prtu(v); }


// #define CHECKNOTEQUAL(x, y)  do { if ((x) != (y)) ; else { \
//     FAIL() <<  "file  " << __FILE__ << " line " << __LINE__ << " error " << #x << " == " << #y << ". Error code " << errno << "\n";\
//     exit(1); \
// } } while (0)
// #define CHECKEQUAL(x, y)  do { if ((x) == (y)) ; else { \
//     FAIL() <<  "file  " << __FILE__ << " line " << __LINE__ << " error " << #x << " != " << #y << ". Error code " << errno << "\n";\
//     exit(1); \
// } } while (0)
// #define CHECKZERO(x)  CHECKEQUAL((x), 0)
// #define CHECKNOTZERO(x)  CHECKNOTEQUAL((x), 0)

#define dprintf(x) do { if (debug < 1) ; else { printf x; fflush(stdout); } } while (0)

#define ALLZEROS(x, K, N) \
do { \
    int allzeros = 1; \
    unsigned xi, xj; \
    for (xi = 0; xi < (unsigned)(K); xi++)      \
	for (xj = 0; xj < (unsigned)(N); xj++)          \
	    allzeros = allzeros & ((x)[xi].v[xj] == 0); \
    if (allzeros) fprintf(stderr, "%s: Unexpected, all %lu elements of %ux%u had all zeros!\n", progname, (unsigned long)K, (unsigned)N, 8/*CHAR_BITS*/*(unsigned)sizeof(x[0].v[0])); \
} while(0)

// /* Read in N words of width W into ARR */
// #define SCANFARRAY(ARR, NAME, N, W) \
// do { \
//     int xi, xj; \
//     unsigned long long xv; \
//     for (xi = 0; xi < (N); xi++) { \
//         /* Avoid any cleverness with SCNx##W because Microsoft (as of Visual Studio 10.x) silently trashes the stack by pretending that %hhx is %x). */ \
// 	const char *xfmt = " %llx%n"; \
// 	ret = sscanf(cp, xfmt, &xv, &xj); \
// 	ARR.v[xi] = (uint##W##_t)xv; \
// 	if (debug > 1) printf("line %d: xfmt for W=%d is \"%s\", got ret=%d xj=%d, %s[%d]=%llx cp=%s", linenum, W, xfmt, ret, xj, #ARR, xi, (unsigned long long) ARR.v[xi], cp); \
// 	if (ret < 1) { \
// 	    fprintf(stderr, "%s: ran out of words reading %s on line %d: " #NAME #N "x" #W " %2d %s", \
// 		    progname, #ARR, linenum, rounds, line); \
// 	    errs++; \
// 	    return; \
// 	} \
// 	cp += xj; \
//     } \
// } while(0)

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


#define LINESIZE 1024

int have_aesni = 0;
int verbose = 0;
int debug = 0;
unsigned nfailed = 0;
char *progname;

extern void host_execute_tests(kat_instance *tests, unsigned ntests);
                
/* A little hack to keep track of the test vectors that we don't know how to deal with: */
int nunknowns = 0;
#define MAXUNKNOWNS 20
const char *unknown_names[MAXUNKNOWNS];
int unknown_counts[MAXUNKNOWNS];

void register_unknown(const char *name){
    int i;
    for(i=0; i<nunknowns; ++i){
        if( strcmp(name, unknown_names[i]) == 0 ){
            unknown_counts[i]++;
            return;
        }
    }
    if( i >= MAXUNKNOWNS ){
        FAIL() << "Too many unknown rng types. Bye.\n";
        exit(1);
    }
    nunknowns++;
    unknown_names[i] = ntcsdup(name);
    unknown_counts[i] = 1;
}

void report_unknowns(){
    int i;
    for(i=0; i<nunknowns; ++i){
        printf("%d test vectors of type %s skipped\n", unknown_counts[i], unknown_names[i]);
    }
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
static int readtest(const char *line, kat_instance* tinst){
    int nchar;
    char name[LINESIZE];
    if( line[0] == '#') return 0;                                       
    sscanf(line, "%s%n", name, &nchar);
    if(!have_aesni){
        /* skip any tests that require AESNI */ 
        if(strncmp(name, "aes", 3)==0 ||
           strncmp(name, "ars", 3)==0){
            register_unknown(name);
            return 0;
        }
    }
#define RNGNxW_TPL(base, N, W) if(strcmp(name, #base #N "x" #W) == 0) return read_##base##N##x##W(line+nchar, tinst);
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

    register_unknown(name);
    return 0;
}

#define RNGNxW_TPL(base, N, W) \
void report_##base##N##x##W##error(const kat_instance *ti){ \
 size_t i;                                                     \
 size_t nkey = sizeof(ti->u.base##N##x##W##_data.ukey.v)/sizeof(ti->u.base##N##x##W##_data.ukey.v[0]); \
 fprintf(stderr, "FAIL:  expected: ");                                \
 fprintf(stderr, #base #N "x" #W " %d", ti->nrounds);                   \
 for(i=0; i<N; ++i){                                                    \
     fprintf(stderr, " "); prtu##W(ti->u.base##N##x##W##_data.ctr.v[i]); \
 }                                                                      \
 for(i=0; i<nkey; ++i){                                                 \
     fprintf(stderr, " "); prtu##W(ti->u.base##N##x##W##_data.ukey.v[i]); \
 }                                                                      \
 for(i=0; i<N; ++i){                                                    \
     fprintf(stderr, " "); prtu##W(ti->u.base##N##x##W##_data.expected.v[i]); \
 }                                                                      \
 fprintf(stderr, "\n");                                                 \
                                                                        \
 fprintf(stderr, "FAIL:  computed: ");                                \
 fprintf(stderr, #base #N "x" #W " %d", ti->nrounds);                   \
 for(i=0; i<N; ++i){                                                    \
     fprintf(stderr, " "); prtu##W(ti->u.base##N##x##W##_data.ctr.v[i]); \
 }                                                                      \
 for(i=0; i<nkey; ++i){                                                 \
     fprintf(stderr, " "); prtu##W(ti->u.base##N##x##W##_data.ukey.v[i]); \
 }                                                                      \
 for(i=0; i<N; ++i){                                                    \
     fprintf(stderr, " "); prtu##W(ti->u.base##N##x##W##_data.computed.v[i]); \
 }                                                                      \
 fprintf(stderr, "\n");                                                 \
 nfailed++;                                                             \
}
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL

// dispatch to one of the report_<GEN>NxW() functions
void analyze_tests(const kat_instance *tests, unsigned ntests){
    unsigned i;
    char zeros[512] = {0};
    for(i=0; i<ntests; ++i){
        const kat_instance *ti = &tests[i];
        switch(tests[i].method){
#define RNGNxW_TPL(base, N, W) case base##N##x##W##_e: \
            if (memcmp(zeros, ti->u.base##N##x##W##_data.expected.v, N*W/8)==0){ \
                fprintf(stderr, "kat expected all zeros?   Something is wrong with the test harness!\n"); \
                nfailed++; \
            } \
            if (memcmp(ti->u.base##N##x##W##_data.computed.v, ti->u.base##N##x##W##_data.expected.v, N*W/8)) \
		report_##base##N##x##W##error(ti); \
	    break;
#include "r123_rngNxW.mm"
#undef RNGNxW_TPL
        case last: ;
        }
    }
}

#define NTESTS 1000

void run() {
    kat_instance *tests;
    unsigned t, ntests = NTESTS;
    char linebuf[LINESIZE];
    FILE *inpfile;
    const char *p;
    const char *inname;
    // char filename[LINESIZE];
    
    progname = "kat tests ";
    inname = "./kat_vectors.txt";
	inpfile = fopen(inname, "r");
	if (inpfile == NULL) {
        FAIL() << "Error opening input file " << inname << " for reading. Received error code " << errno << "\n";
	    exit(1);
	}
    if ((p = getenv("KATC_VERBOSE")) != NULL) {
	    verbose = atoi(p);
    }
    if ((p = getenv("KATC_DEBUG")) != NULL) {
	    debug = atoi(p);
    }

#if R123_USE_AES_NI
    have_aesni = haveAESNI();
#else
    have_aesni = 0;
#endif

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
        if( readtest(linebuf, &tests[t]) )
            ++t;
    }
    if(t==ntests){
	    FAIL() << "No more space for tests?  Recompile with a larger NTESTS\n";
	    exit(1);
    }
    tests[t].method = last; // N.B  *not* t++ - the 'ntests' value passed to host_execute_tests does not count the 'last' one.

    report_unknowns();
    printf("Perform %lu tests.\n", (unsigned long)t);
    host_execute_tests(tests, t);

    analyze_tests(tests, t);
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

class TestRandom123 : public ::testing::Test { };

TEST_F(TestRandom123, kat) {
    run();
}
