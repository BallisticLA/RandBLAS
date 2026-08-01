# Native counter-based RNG design

Date: 2026-07-31

Updated: 2026-08-01

## Summary

RandBLAS will replace its Random123 dependency with native, header-only Philox
and floating-point transformation implementations. The public integer-generator
boundary will be a stateless block function with output written indirectly:

```cpp
void generate(ctr_t const& counter,
              key_t const& key,
              res_t& output) const;
```

RandBLAS algorithms will consume state-like objects. The provided
`RNGState<Engine>` will adapt a stateless counter-based engine to that interface,
while `RepackedOutput<Engine, OutputWord>` will expose the same random block as a
larger number of narrower output words.

This change will ship Philox only. It will not ship Squares or another modern
post-Random123 generator while the licensing of the Squares reference material
is unresolved. The engine, counter, seed-mapping, and output-adaptor boundaries
will nevertheless allow a future Squares or Squares-like engine to be added
without changing RandBLAS samplers or `RNGState`.

The migration will land atomically. RandBLAS source, tests, examples,
documentation, CI, installation, and installed CMake packages will all work when
Random123 is absent.

## Goals

- Remove Random123 as a build, install, test, and transitive package dependency.
- Provide `RandBLAS::rng::Philox<N, W, R>` for every parameter combination
  Random123 supports:
  - `N` equal to 2 or 4;
  - `W` equal to 32 or 64; and
  - `R` from 0 through 16, inclusive.
- Produce exactly the same integer block as Random123 for the same valid Philox
  parameters, counter, and key.
- Preserve the current default stream through `Philox<4, 32, 10>`.
- Preserve bitwise output of default-engine sparse sketching operators.
- Preserve dense-sketch reproducibility up to IEEE-compliant rounding of
  `sin`, `cos`, `log`, and `sqrt`; cross-platform bitwise identity of dense
  sketches is not required.
- Preserve thread-count-independent, coordinate-addressable sampling and the
  existing state-advance rules.
- Expose structural C++20 engine and state concepts without inheritance or
  virtual dispatch.
- Implement and test `RepackedOutput` for power-of-two subdivisions of native
  output words.
- Make counter advancement, key construction, and result shape engine-owned
  choices so a future Squares-like engine does not require changes to generic
  RandBLAS code.
- Match current performance within normal benchmark variation on supported
  platforms.
- Document the RNG design and its relationship to the C++ standard random
  facilities in developer notes before merge.

## Non-goals

- Squares, Collatz-Weyl generators, or any other post-Random123 generator in
  this change.
- Partial-width modular counter arithmetic not used by Philox. A future engine
  that needs an early wrap period will provide its own `ctr_t`.
- Native Threefry, `MicroURNG`, `Engine`, AES, ARS, or other Random123 APIs.
- Requiring every current RandBLAS sampler to consume 8- or 16-bit result words.
  `RepackedOutput` is implemented at the engine and state levels in this change;
  broader low-precision sampling is future work.
- Modeling `std::uniform_random_bit_generator` or providing a scalar STL engine
  adaptor.
- CUDA device execution. Native RNG headers need only compile in host code when
  processed by NVCC or included with a CUDA-aware BLAS++ configuration.
- Bitwise equality of dense sketches across math libraries, compilers, or
  architectures.
- A cryptographically secure built-in RNG.
- A Random123/native build switch or an implementation in namespace `r123`.
- RandLAPACK migration. RandLAPACK will adapt separately to the API selected by
  RandBLAS.
- Performance work beyond matching the current implementation.

## Considered approaches

### State-like public concept with a stateless-engine adapter (selected)

RandBLAS algorithms template on a state that generates the current block and
advances by blocks. `RNGState<Engine>` adapts a stateless counter-based engine to
this interface.

This matches what RandBLAS algorithms consume, hides engine-specific counter and
key representations, and preserves inexpensive coordinate-addressed sampling.

### Engine-like public concept

This would expose the counter, key, and engine separately throughout RandBLAS.
It would resemble Random123 but would make an implementation detail the primary
customization contract. It was rejected because the state interface is smaller
and closer to sampler behavior.

### Distribution-aware random source

This would make one policy responsible for integer generation, uniforms, and
Gaussians. It was rejected because it would couple the counter-based engine
contract to the current transformation algorithms.

### Counter-owned advancement (selected)

Every engine defines a copyable `ctr_t` with `advance(uint64_t)`. `RNGState`
delegates modular advancement to that value type. Philox uses a full-width word
array; a future Squares-like engine can use a counter with a different logical
width and wrap period.

### Universal partial-width counter

This would implement one counter template parameterized by storage width and
logical width. It was rejected for this change because Philox needs only
full-width arithmetic. Partial-width arithmetic will arrive with an engine that
uses and tests it.

### Engine-owned counter advancement

This would require each engine to provide a static operation that mutates its
counter. It was rejected because advancement is an integer value-type behavior,
and placing it on the engine would force adaptors to forward more engine-specific
operations.

### Repacked output as an engine adaptor (selected)

`RepackedOutput<Engine, OutputWord>` changes only the representation of an
engine's result block. It preserves counter, key, seed mapping, counter period,
and block advancement. This keeps low-precision output policy independent of
Philox and makes it reusable with future engines.

Adding an output-width parameter directly to `Philox` was rejected because it
would conflate the underlying generator with a representation of its output and
would not transfer to other engines.

## Source organization

The implementation will be split into focused headers:

- `RandBLAS/rng/word_array.hh` provides full-width fixed-word storage and the
  carry arithmetic used by Philox counters and scalar seed-to-key mapping.
- `RandBLAS/rng/philox.hh` provides the stateless Philox engine.
- `RandBLAS/rng/repacked_output.hh` provides the result-word adaptor.
- `RandBLAS/rng/distributions.hh` provides the retained integer-to-floating
  conversions and Box--Muller transformation.
- `RandBLAS/random_gen.hh` remains the public umbrella and provides concepts,
  `RNGState<Engine>`, default aliases, and native implementation includes.
- `RandBLAS/rng/DevNotes.md` records algorithm provenance, the relationship to
  the standard library, seed semantics, output ordering, testing strategy, and
  the process for adding an engine. The existing RandBLAS developer notes will
  link to this file.

RNG state definitions currently in `RandBLAS/base.hh` may move into
`RandBLAS/random_gen.hh` so the RNG subsystem has one entry point. `base.hh`
will continue to make default RNG types available through its inclusion of
`random_gen.hh`.

## Stateless engine contract

A counter-based engine exposes `ctr_t`, `key_t`, and `res_t` and provides:

```cpp
void generate(ctr_t const& counter,
              key_t const& key,
              res_t& output) const;
```

`generate` writes every output element and returns `void`. Its third argument is
output-only and is distinct from the input counter. The counter and key are not
mutated.

The engine contract is structural. The conceptual C++20 requirement is:

```cpp
template <class Engine>
concept CounterBasedEngine =
    requires(Engine const& engine,
             typename Engine::ctr_t const& counter,
             typename Engine::key_t const& key,
             typename Engine::res_t& output) {
        typename Engine::ctr_t;
        typename Engine::key_t;
        typename Engine::res_t;
        { engine.generate(counter, key, output) } -> std::same_as<void>;
    };
```

The final concept will also check the value semantics, fixed result extent,
unsigned result words, and counter advancement required by `RNGState`. It will
check expressions rather than require a particular class identity.

An engine may optionally define:

```cpp
static key_t make_key(uint64_t seed);
```

This hook owns the interpretation of a scalar seed. It keeps generic code from
assuming that arbitrary bit patterns are valid keys. `RNGState(uint64_t)` exists
only when its engine supports this hook. Explicit raw-key construction remains
available for known-answer tests and advanced use. An engine adaptor forwards
`make_key` when its wrapped engine provides it.

Engine types have value semantics and require no polymorphic base. Integer-only
operations will be `constexpr` and `noexcept` where their underlying operations
permit it.

## Native Philox

The native engine has the public form:

```cpp
RandBLAS::rng::Philox<N, W, R>
```

For a valid specialization it exposes `ctr_t`, `key_t`, and `res_t`. A
default-constructed engine writes one counter-sized result block without
storing or mutating random state.

The implementation preserves Random123's:

- counter and key word ordering, with word zero treated as least significant;
- multiplication and key-bump constants;
- high/low multiplication behavior;
- round order, permutations, and XOR operations; and
- modular unsigned arithmetic.

`Philox::make_key(seed)` preserves the current `RNGState(seed)` meaning: begin
with a zero key and increment it by `seed` using the key's extended-width
unsigned interpretation. The counter begins at zero.

Invalid `N`, `W`, or `R` values produce clear compile-time diagnostics. The
32-bit variants use 64-bit multiplication. The 64-bit variants use portable
mechanisms for the supported GNU, Clang, Apple Clang, and MSVC matrix and do not
introduce broader platform requirements.

## Counter value types and advancement

An engine's `ctr_t` is a copyable value type with:

```cpp
void advance(uint64_t blocks);
```

Advancement is modular according to that counter type. `RNGState` neither
inspects its storage nor assumes that the logical counter width equals the
storage width.

Philox uses a fixed-size array of unsigned 32- or 64-bit words. It supports value
initialization to zero, indexed const observation, equality, copying, and
carry-propagating advancement from lower- to higher-indexed words. Overflow of
the most-significant word wraps.

This change does not implement a general partial-width counter. A future
`Squares<N>` could define a `ctr_t` whose logical width is
`64 - log2(N)` and whose `advance` wraps at that width without altering
`RNGState`, `RepackedOutput`, or any sampler.

Counters and keys may be exposed by const accessors for construction, testing,
and diagnostics. Mutable storage is not part of either public concept.

## State-like customization boundary

RandBLAS defines a documented `CounterBasedRNGState` concept. A conforming state
is copyable and provides a fixed-size `res_t` of unsigned words plus:

```cpp
void generate(res_t& output) const;
void advance(uint64_t blocks);
```

The concept does not require public counters, keys, or engines.

`RNGState<Engine>` stores the engine's `ctr_t` and `key_t` and a
`[[no_unique_address]] Engine`. It follows the Rule of Zero. `generate` delegates
to the engine without mutation; `advance` delegates to `ctr_t::advance`.

`RNGState` provides:

- value initialization when the engine's counter and key support it;
- construction from an explicit key with a zero counter;
- construction from explicit counter and key values;
- scalar-seed construction only when `Engine::make_key` exists; and
- const counter and key observation where retained for migration and debugging.

The default aliases are equivalent to:

```cpp
using DefaultRNG = rng::Philox<4, 32, 10>;
using DefaultRNGState = RNGState<DefaultRNG>;
```

`RNGState<>` remains shorthand for the default state.

## `RepackedOutput`

The adaptor has the public form:

```cpp
RandBLAS::rng::RepackedOutput<Engine, OutputWord>
```

It aliases the wrapped engine's `ctr_t` and `key_t`, defines a new `res_t`, and
preserves the total number of bits in a result block. `OutputWord` must be an
unsigned integer whose bit width divides the native result-word width by a
power-of-two ratio.

Initial support includes direct or nested:

- 32-bit words to 16-bit words;
- 32-bit words to 8-bit words; and
- 16-bit words to 8-bit words.

Native result-word order is preserved. Within each native word, chunks appear
from least significant to most significant, independent of host endianness. For
example, repacking `0xAABBCCDD` yields `{0xCCDD, 0xAABB}` as 16-bit words and
`{0xDD, 0xCC, 0xBB, 0xAA}` as 8-bit words.

`generate` creates native local storage, asks the wrapped engine to fill it, and
then fills the adapted output array with shifts and masks. The adaptor forwards
`make_key` when available. It does not define new counter behavior: its `ctr_t`
is the wrapped type, so period and `advance(1)` retain native block semantics.

This change does not require existing samplers to accept 8- or 16-bit words.
Operations may impose additional word-width or result-length constraints with
clear compile-time diagnostics. The retained metadata and bit ordering make
future low-precision sampling possible without changing the underlying stream.

## RandBLAS API migration

Sketching operators and sampling functions template on state types rather than
stateless engines. For example, the conceptual dense operator becomes:

```cpp
template <typename T, CounterBasedRNGState State = DefaultRNGState>
struct DenseSkOp;
```

The same rule applies to sparse operators, dense and sparse fill functions,
index sampling, testing helpers, and entry points that currently propagate an
`RNG` parameter. Stored `seed_state` and `next_state` members have type `State`
directly.

The base state concept remains small. Individual algorithms may impose further
requirements. For example, a sparse sampler that consumes four result words may
require at least four suitably wide words. It will not silently consume an
unspecified number of additional blocks.

No compatibility types are defined in namespace `r123`. Existing `r123ext`
helpers move into `RandBLAS::rng`. Cheap RandBLAS-native aliases may be retained
where they improve migration without obscuring the new API.

## Sampling data flow

Sampling remains coordinate-addressable rather than sequentially dependent on
thread scheduling:

1. A function accepts a state by const reference.
2. It copies the state for each independent region or worker.
3. It computes a block offset from dimensions, distribution layout, and matrix
   coordinates.
4. It calls `advance(offset)` on the local copy.
5. It calls `generate(output)` into a local `res_t` and transforms those words.
6. It returns a copied state advanced by the total number of reserved blocks.

Dense sampling preserves current row padding and block-address mapping. Sparse
sampling preserves its current reservation of one default-engine block per
nonzero. These rules preserve input-state nonmutation, OpenMP thread-count
independence, full/submatrix consistency, existing `next_state` values, and
default-engine sparse output bits.

## Relationship to the C++ standard library

RandBLAS uses the term *counter-based engine* differently from the C++ standard
random-number-engine requirement.

A standard engine is a stateful scalar uniform-random-bit generator. It exposes
mutating `operator()`, scalar `result_type`, seeding, serialization, and
`discard`. The standard `philox_engine` also stores a counter, key, cached result
block, and index into that block so it can return one scalar word per call.

A RandBLAS engine is instead a stateless block function from `(counter, key)` to
`res_t`. `RNGState<Engine>` supplies only the stateful operations RandBLAS needs:
nonmutating block generation and explicit block advancement. `RepackedOutput`
is a block-level adaptor, not a standard random-engine adaptor.

Neither RandBLAS engine nor state concepts model
`std::uniform_random_bit_generator` in this change. A future scalar adaptor can
be added independently. Using that interface internally now would obscure block
boundaries and coordinate-addressed sampling.

The permanent RNG developer notes will include this comparison. The temporary
implementation plan will also compare each RandBLAS base RNG abstraction to its
nearest standard-library counterpart before implementation tasks begin.

## Floating-point transformations

RandBLAS faithfully adapts the Random123 formulas it uses rather than switching
to standard-library distributions or a different normal transform. The native
implementation preserves:

- `u01` endpoint and scaling behavior;
- `uneg11` endpoint and scaling behavior;
- each other conversion still used by equivalent RandBLAS functionality;
- the Box--Muller assignment of words to angle and radius;
- sine and cosine output order; and
- constants and default output precision.

As at present, 32-bit words produce `float` samples and 64-bit words produce
`double` samples, followed by promotion to the matrix scalar type. Existing
sampling operations may reject narrower words until their bit-assembly policy is
designed.

The host implementation uses `std::sin`, `std::cos`, `std::log`, and
`std::sqrt`. This preserves the mathematical mapping but does not promise
cross-platform bitwise identity. Standard-library random distributions are not
used because their exact mappings and engine-consumption patterns are not
portable, and some distributions cache results or consume a variable number of
engine values.

## Error handling

- Invalid Philox template parameters fail at compile time.
- Malformed engine and state types fail at their concept boundaries.
- Invalid repacking word types or ratios fail at compile time.
- Operation-specific word-width and result-length requirements fail at compile
  time.
- Scalar seed construction is absent when an engine has no `make_key` hook.
- Counter and key arithmetic uses defined unsigned modular behavior.
- Existing dimension, buffer, and checked-product validation remains in place.
- There is no runtime backend selection or new RNG-specific exception path.

## Build, installation, and CI

The atomic migration removes Random123 from:

- top-level `find_package` calls;
- interface libraries and include paths;
- `cmake/FindRandom123.cmake`;
- installed `RandBLASConfig.cmake` dependency discovery and cached paths;
- example build definitions;
- CI dependency setup, caches, inputs, and environment variables;
- downstream package-consumer configurations; and
- installation instructions.

An installed package must configure and compile a consumer without Random123.
Existing host-build coverage for CUDA-aware BLAS++ and NVCC-parsed headers
remains; native RNG code does not add CUDA device annotations or device math.

## Test design

The inherited `test/basic_rng/test_r123.cc` is rewritten around native RandBLAS
functionality and may be renamed `test_philox.cc`. Tests whose only purpose is
Threefry, `MicroURNG`, `Engine`, or another unsupported Random123 facility are
removed.

### Philox, counter, and state tests

- Preserve all applicable published Philox known-answer vectors already in the
  repository.
- Generate additional vectors once, offline from the pinned Random123 checkout,
  so every `N` in `{2,4}`, `W` in `{32,64}`, and `R` in `[0,16]` has direct
  coverage. Checked-in tests use static data and never locate Random123.
- Verify that `generate` fills its output and does not mutate counter or key.
- Test zero, single-word carry, multiword carry, large advancement, and full
  modular wraparound.
- Test default construction, raw counter/key construction, scalar seed
  compatibility, copying, equality where retained, nonmutating generation, and
  state advancement.
- Add compile-time assertions for the default engine and state concepts.
- Add a test-only engine with an opaque, non-Philox, full-width counter type to
  prove `RNGState` delegates generation, key mapping, and advancement without
  inspecting representations.

### `RepackedOutput` tests

- Test direct `32->16`, `32->8`, and `16->8` repacking.
- Test nested adaptors and equality with equivalent direct repacking.
- Test native word order and least-significant-chunk-first order with fixed
  hexadecimal values.
- Verify endian-independent expected results.
- Verify preservation of total block bits, `ctr_t`, `key_t`, `make_key`, and
  block advancement.
- Verify an adapted `RNGState` produces the repacked bits of the same native
  block and advances by the same number of blocks.
- Add compile-time checks rejecting signed, wider, non-dividing, and
  non-power-of-two output widths.

### Distribution and sampler tests

- Adapt endpoint and reference tests for retained integer-to-floating
  conversions.
- Test Box--Muller results with tolerances appropriate for host math libraries.
- Retain continuous and discrete statistical tests.
- Retain dense and sparse state-advance, thread-count-independence, and
  full/submatrix consistency tests.
- Retain deterministic sparse-operator expectations with the default state.
- Retain tests of all public sketching APIs after migrating engine template
  parameters to state types.

### Package tests

- Configure and build RandBLAS without a Random123 path or installation.
- Install RandBLAS and build the downstream consumer against the installed
  package.
- Install RandBLAS and build the examples without Random123.
- Exercise the supported compiler and CI matrix, including CUDA-aware host
  compilation.

## Performance validation

Before implementation, run the current basic RNG benchmark and relevant dense
and sparse sampling benchmarks under the workspace's Spack environment. After
migration, rerun equivalent native benchmarks with the same toolchain and
settings. A visible regression outside ordinary run-to-run variation must be
investigated. Optimization beyond parity requires a separate proposal and
before/after benchmark evidence.

## Documentation, provenance, and licensing

Directly adapted Philox source, floating-point transformations, and test material
retain applicable D. E. Shaw Research copyright and BSD-3-Clause notices. The
developer notes identify adapted Random123 algorithms and vectors and cite the
Philox paper.

No Squares or Collatz-Weyl source is incorporated. A future Squares-like engine
requires a separate design and a BSD-compatible implementation basis. The
current design makes no commitment to a Squares key mapping. Any future mapping
from `uint64_t` seeds to constrained Squares keys must be stable, documented,
and covered by inter-key statistical tests.

User and API documentation explains:

- `DefaultRNGState` and `RNGState<>`;
- `CounterBasedEngine` and `CounterBasedRNGState`;
- `ctr_t`, `key_t`, and `res_t`;
- output-only block generation;
- engine-owned counter advancement and scalar seed mapping;
- `RepackedOutput` bit order and block semantics;
- exact Philox integer-stream compatibility;
- floating-point reproducibility boundaries;
- coordinate-addressed, thread-independent sampling;
- differences from the C++ standard random facilities; and
- the non-cryptographic nature of native Philox.

Historical references to Random123 remain only where they provide attribution,
provenance, or migration context. Installation documentation no longer describes
Random123 as a dependency.

## Rollout

Native implementation and dependency removal land atomically. There is no
compatibility window, feature flag, or dual backend. Small RandBLAS-native aliases
or adaptors may ease migration, but Random123 names and headers do not remain in
the public API.

RandLAPACK changes are a separate follow-up and do not constrain this design.

## Acceptance criteria

The work is complete when all of the following hold:

1. `Philox<N, W, R>` passes native known-answer tests for every supported
   template combination.
2. The default native engine matches Random123 integer blocks for the same
   counter and key.
3. Engine and state generation use the approved output-only `res_t&` APIs.
4. `RNGState` works with the test-only non-Philox counter representation without
   generic code inspecting it.
5. `RepackedOutput` passes direct, nested, ordering, forwarding, advancement,
   and compile-time rejection tests.
6. Default-engine sparse sketches remain bitwise unchanged.
7. Dense transforms preserve the Random123 mathematical mapping within the
   stated floating-point boundary.
8. Thread-count independence, full/submatrix equivalence, and state-advance
   invariants pass.
9. RandBLAS configures, builds, and passes the full Spack-based test suite with
   no Random123 installation.
10. Installed-package consumers and examples build without Random123.
11. Supported CI configurations, including CUDA-aware host builds, pass.
12. Before/after benchmarks show no material regression.
13. RNG developer notes document the design, provenance, STL comparison, and
    future-engine extension points.
14. Build files, package metadata, CI, examples, and current documentation
    contain no functional Random123 dependency; remaining references are limited
    to attribution, provenance, or historical context.
