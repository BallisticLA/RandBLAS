# Native counter-based RNG design

Date: 2026-07-31

## Summary

RandBLAS will replace its Random123 dependency with a native, header-only
implementation of the Philox counter-based random number generator (CBRNG) and
the floating-point transformations RandBLAS uses. The integer generator will be
a faithful, trimmed adaptation of Random123 and will retain its attribution and
license notices.

The public customization boundary will be a state-like C++20 concept. RandBLAS
will provide a generic `RNGState<Engine>` adapter for stateless CBRNG engines and
will use `RNGState<rng::Philox<4, 32, 10>>` by default. Sketching operators and
sampling functions will template on the state type rather than the engine type.

The change will land atomically. RandBLAS's source, tests, examples,
documentation, CI, installation, and installed CMake package will all work when
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
- Preserve the existing default random stream by using
  `Philox<4, 32, 10>` as the default engine.
- Preserve bitwise output of default-engine sparse sketching operators.
- Preserve dense-sketch reproducibility up to IEEE-compliant rounding of
  `sin`, `cos`, `log`, and `sqrt`; cross-platform bitwise identity of dense
  sketches is not required.
- Preserve thread-count-independent, coordinate-addressable sampling and the
  existing state-advance rules.
- Expose a state-like C++20 concept that third-party CBRNG states, including a
  future cryptographically secure implementation, can satisfy.
- Match the performance of the current implementation within normal benchmark
  variation on the current supported platforms.

## Non-goals

- Native Threefry, `MicroURNG`, `Engine`, AES, ARS, or other Random123 APIs.
- CUDA device execution. Native RNG headers need only compile in host code when
  processed by NVCC or included with a CUDA-aware BLAS++ configuration.
- Bitwise equality of dense sketches across different math libraries,
  compilers, or architectures.
- A cryptographically secure built-in RNG.
- A Random123/native build-time switch or a compatibility implementation in the
  `r123` namespace.
- RandLAPACK migration. RandLAPACK will adapt separately to the API selected by
  RandBLAS.
- Performance improvements beyond matching the current implementation.

## Considered approaches

### State-like public concept with a stateless-engine adapter (selected)

RandBLAS algorithms template on a state type that can generate the current
integer block and advance by a number of blocks. `RNGState<Engine>` adapts a
conventional stateless CBRNG engine to this interface.

This matches what RandBLAS algorithms actually consume, hides counter and key
representation from generic algorithms, preserves inexpensive random access,
and lets already-stateful generators integrate without mimicking Random123's
engine API.

### Engine-like public concept

This would preserve the current `RNGState<RNG>`-centered template structure and
minimize source edits, but it would expose Random123's separation of engine,
counter, and key as RandBLAS's primary customization contract. It was rejected
because state generation and advancement are the smaller and more natural
RandBLAS interface.

### Distribution-aware random source

This would make one policy responsible for integer generation, uniforms, and
Gaussians. It could support alternate distribution algorithms, but it would
couple the CBRNG contract to RandBLAS's current floating-point transforms and
would introduce unnecessary policy machinery. It was rejected.

## Source organization

The RNG implementation will be split into focused headers:

- `RandBLAS/rng/word_array.hh` provides the fixed-width unsigned-word storage
  and carry-propagating advancement needed for counters and keys.
- `RandBLAS/rng/philox.hh` provides the stateless Philox engine.
- `RandBLAS/rng/distributions.hh` provides the retained Random123-compatible
  integer-to-floating-point conversions and Box--Muller transformation.
- `RandBLAS/random_gen.hh` remains the public umbrella header. It provides the
  state concept, the `RNGState<Engine>` adapter, default aliases, and includes
  the native implementation headers.

RNG state definitions currently located in `RandBLAS/base.hh` may move into
`RandBLAS/random_gen.hh` so that the RNG subsystem has one clear entry point.
`base.hh` will continue to make the default RNG types available through its
existing inclusion of `random_gen.hh`.

## Stateless Philox engine

The native engine will have the public form:

```cpp
RandBLAS::rng::Philox<N, W, R>
```

For a valid specialization it will expose unsigned `word_type`, fixed-size
`counter_type`, `key_type`, and `result_type` aliases. Calling a default-
constructed engine with a counter and key will return one counter-sized result
block without storing or mutating state.

The implementation will preserve Random123's:

- counter and key word ordering, with word zero treated as least significant;
- multiplication constants and key-bump constants;
- high/low multiplication behavior;
- round order, permutations, and XOR operations; and
- modular unsigned arithmetic.

Invalid values of `N`, `W`, or `R` will produce a clear compile-time diagnostic.
The 32-bit variants will use 64-bit multiplication. The 64-bit variants will use
the portable mechanisms required by the current GNU, Clang, Apple Clang, and
MSVC support matrix. The implementation will not introduce broader platform
requirements than current RandBLAS.

## Word arrays and advancement

Native counter and key storage will be fixed-size arrays of unsigned 32- or
64-bit words. The type will support:

- value initialization to zero;
- indexed access and fixed compile-time size;
- equality and copying; and
- advancing by a 64-bit amount with carry propagation from lower to higher
  indexed words.

Advancement is modular. Overflow of the most-significant word wraps rather than
raising an error, matching Random123's behavior. RandBLAS's existing dimension
and safe-integer-product validation remains responsible for detecting invalid
sampling sizes before state advancement is computed.

Generic RandBLAS algorithms will not depend on the word-array representation.
The provided adapter may expose counter and key values for construction,
testing, or debugging, but such access is not part of the state concept.

## State-like customization boundary

RandBLAS will define a documented `CounterBasedRNGState` concept. A conforming
state must be copyable and provide:

- a fixed-size, indexable `result_type` whose element type is an unsigned
  integer;
- the result block size as a compile-time value;
- `generate() const`, returning the block at the current state without mutation;
  and
- `advance(uint64_t blocks)`, advancing by that many result blocks.

The precise spelling of the compile-time block-size member may follow existing
RandBLAS conventions, but the concept will not require public counter, key, or
engine members.

`RNGState<Engine>` will be RandBLAS's adapter for a stateless engine. It will
store the engine's counter and key, implement `generate()` by evaluating the
engine at those values, and implement `advance()` through carry-propagating
counter advancement. Its seed constructor will preserve the current meaning:
the counter starts at zero and the supplied integer initializes the key by
advancing a zero key.

The default aliases will be equivalent to:

```cpp
using DefaultRNG = rng::Philox<4, 32, 10>;
using DefaultRNGState = RNGState<DefaultRNG>;
```

`RNGState<>` will remain shorthand for the default-engine state.

## RandBLAS API migration

Sketching operators and sampling functions will template on state types rather
than stateless engines. For example, the conceptual form of the dense operator
will become:

```cpp
template <typename T, CounterBasedRNGState State = DefaultRNGState>
struct DenseSkOp;
```

The same rule applies to `SparseSkOp`, dense and sparse fill functions, index
sampling utilities, testing helpers, and sketching entry points that currently
propagate an `RNG` template parameter. Their stored `seed_state` and
`next_state` members will have type `State` directly.

The base state concept is intentionally small. Individual algorithms may impose
additional compile-time requirements. Current sparse sampling uses four words
from each generated block, so sparse operations that write indices and signs
will require a result block of at least four words. They will not silently
consume multiple two-word blocks. Consequently `Philox<2, W, R>` remains a
supported engine for integer generation, dense sampling, and any compatible
operation, while unsupported sparse uses fail with a clear compile-time
diagnostic.

No compatibility types will be defined in namespace `r123`. The old
`r123ext` helpers will move into `RandBLAS::rng`. Cheap RandBLAS-native aliases
may be retained where they improve migration without obscuring the new API.

## Sampling data flow

Sampling remains coordinate-addressable rather than sequentially dependent on
thread scheduling:

1. A sampling function accepts a state by const reference.
2. It copies that state for each independent region or worker.
3. It computes a block offset solely from matrix dimensions, distribution
   layout, and requested matrix coordinates.
4. It calls `advance(offset)` on the local copy.
5. It calls `generate()` and transforms the integer block into the requested
   scalar or discrete values.
6. It returns a copied state advanced by the total number of blocks reserved by
   the operation.

Dense sampling will preserve its current row padding and block-address mapping.
Sparse sampling will preserve its current rule of reserving one default-engine
block per nonzero. These rules preserve:

- independence from OpenMP thread count;
- consistency between full-matrix and submatrix generation;
- nonmutation of input states;
- existing `next_state` values; and
- default-engine sparse operator bits.

## Floating-point transformations

RandBLAS will faithfully adapt the Random123 formulas it uses rather than switch
to standard-library distributions or a different normal transform. The native
implementation will preserve:

- `u01` endpoint and scaling behavior;
- `uneg11` endpoint and scaling behavior;
- any other conversion still used by equivalent RandBLAS functionality;
- the Box--Muller assignment of input words to angle and radius;
- the order of sine and cosine outputs; and
- the constants and default output precision.

As in the current implementation, 32-bit generator words produce `float`
samples and 64-bit words produce `double` samples. Promotion to the matrix
scalar type happens afterward.

The host implementation will call `std::sin`, `std::cos`, `std::log`, and
`std::sqrt`. This preserves the mathematical mapping and provides
reproducibility up to compliant floating-point rounding, but does not promise
cross-platform bitwise identity. Standard-library random distributions will not
be used because their exact mappings and engine-consumption patterns are not
portable, and normal distributions may cache values or consume a variable
number of engine results.

## Error handling

- Invalid Philox template parameters fail at compile time.
- Types that do not satisfy `CounterBasedRNGState` fail at compile time at the
  API boundary.
- Operation-specific block-size requirements fail at compile time.
- Counter and key arithmetic uses defined unsigned modular behavior.
- Existing RandBLAS runtime validation for dimensions, buffer requirements, and
  checked integer products remains in place.
- There is no runtime backend selection and no new RNG-specific exception path.

## Build, installation, and CI changes

The atomic migration will remove Random123 from:

- the top-level `find_package` calls;
- RandBLAS interface libraries and include paths;
- `cmake/FindRandom123.cmake`;
- installed `RandBLASConfig.cmake` dependency discovery and cached paths;
- example build definitions;
- CI dependency setup, caches, inputs, and environment variables;
- downstream package-consumer configurations; and
- installation instructions.

The installed package must configure and compile a consumer without Random123
present. Existing host-build coverage for CUDA-aware BLAS++ and NVCC-parsed
headers will remain, but the new implementation will not add CUDA device
annotations or device math paths.

## Test design

The inherited `test/basic_rng/test_r123.cc` will be rewritten around native
RandBLAS functionality and may be renamed `test_philox.cc`. Its broad
Random123-specific harness will be replaced with focused GoogleTest cases.

### Philox and state tests

- Preserve all applicable published Philox known-answer vectors already in the
  repository.
- Generate additional vectors once, offline from the pinned Random123 checkout,
  so each combination of `N` in `{2,4}`, `W` in `{32,64}`, and `R` in
  `[0,16]` has direct known-answer coverage. The checked-in tests will consume
  only static vector data and will not invoke or locate Random123.
- Test zero, single-word carry, multiword carry, large advancement, and full
  modular wraparound.
- Test seed construction, copying, equality where retained, nonmutating
  generation, and returned state advancement.
- Add compile-time assertions for `DefaultRNGState` and a small custom state that
  satisfies `CounterBasedRNGState`.
- Test that the provided adapter produces the same block as invoking its engine
  at the adapter's counter and key.

Tests whose only purpose is Random123 Threefry, `MicroURNG`, `Engine`, or another
unsupported facility will be removed.

### Distribution and sampling tests

- Adapt reference and endpoint tests for each retained integer-to-floating
  conversion.
- Test Box--Muller results with floating-point tolerances appropriate for host
  math-library rounding.
- Retain the existing continuous and discrete statistical tests.
- Retain dense and sparse state-advance tests.
- Retain thread-count-independence tests.
- Retain full-matrix/submatrix consistency tests.
- Retain deterministic sparse-operator expectations with the default state.
- Retain tests of all public sketching APIs after migrating their template
  parameters from engine types to state types.

### Package tests

- Configure and build RandBLAS without a Random123 path or installation.
- Install RandBLAS and build the existing downstream consumer against the
  installed package.
- Install RandBLAS and build the examples without Random123.
- Exercise the current supported compiler and CI matrix, including host
  compilation in CUDA-aware configurations.

## Performance validation

Before implementation changes, run the current basic RNG benchmark and relevant
dense and sparse sampling benchmarks under the workspace's Spack environment.
After the migration, rerun the same binaries or equivalent native versions with
the same toolchain and settings. A visible regression outside ordinary run-to-
run variation must be investigated. Optimization beyond parity requires a
separate proposal and before/after benchmarking.

## Documentation and attribution

Directly adapted source and test material will retain the applicable D. E. Shaw
Research copyright and BSD-3-Clause license notice. Developer notes will identify
the adapted Random123 algorithms and vectors and cite the Philox paper.

User and API documentation will explain:

- `DefaultRNGState` and `RNGState<>`;
- the `CounterBasedRNGState` customization contract;
- how `RNGState<Engine>` adapts a stateless CBRNG;
- exact Philox integer-stream compatibility;
- floating-point reproducibility boundaries;
- coordinate-addressed, thread-independent sampling; and
- the non-cryptographic nature of the built-in Philox engine.

Historical or attribution comments that name Random123 will remain where they
provide specific context. Installation and usage documentation will no longer
describe Random123 as a dependency.

## Rollout

The native implementation and dependency removal will land atomically. There
will be no compatibility window, feature flag, or dual backend. The change may
use small RandBLAS-native aliases or adapters, but Random123 names and headers
will not remain part of the public API.

RandLAPACK changes are a separate follow-up and do not constrain this design.

## Acceptance criteria

The work is complete when all of the following hold:

1. `Philox<N, W, R>` passes native known-answer tests for all supported template
   parameter combinations.
2. The default native engine matches Random123's integer blocks exactly for the
   same counter and key.
3. Default-engine sparse sketches remain bitwise unchanged.
4. Dense transforms preserve the Random123 mathematical mapping within the
   stated floating-point reproducibility boundary.
5. Thread-count independence, full/submatrix equivalence, and state-advance
   invariants pass their tests.
6. RandBLAS configures, builds, and passes the full test suite using the
   workspace's Spack environment without Random123 present.
7. An installed-package consumer and the RandBLAS examples build without
   Random123.
8. Current supported CI configurations, including CUDA-aware host builds, pass.
9. The before/after benchmarks show no material regression.
10. Build files, installed package metadata, CI, examples, and current
    documentation contain no functional Random123 dependency; remaining
    references are limited to attribution, provenance, or historical context.
