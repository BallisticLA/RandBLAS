# Random-number generation developer notes

RandBLAS is migrating from Random123 to native, header-only counter-based
random-number generation. The target API and invariants are recorded here while
the implementation is in progress. Until the migration commit lands,
`RandBLAS/random_gen.hh` and `RandBLAS/base.hh` still expose the Random123-backed
implementation.

## Public engine and state contracts

A counter-based engine is a stateless block function. It owns the types and
meaning of its counter, key, and result:

```cpp
void generate(ctr_t const& counter,
              key_t const& key,
              res_t& output) const;
```

The third argument is output-only. The engine writes every lane and does not
mutate the counter or key. The engine contract is structural; engines do not
inherit from a RandBLAS base class.

RandBLAS algorithms consume state-like objects instead of engines directly. A
state provides a fixed-size unsigned `res_t` and these operations:

```cpp
void generate(res_t& output) const;
void advance(std::uint64_t blocks);
```

`RNGState<Engine>` adapts an engine to that boundary by storing its counter,
key, and an empty engine value. Algorithms are generic over the state contract
and do not inspect those stored representations.

## Relationship to the C++ standard random facilities

| RandBLAS abstraction | Nearest standard-library abstraction | Deliberate difference |
|---|---|---|
| `rng::WordArray<Word, N>` | `std::array<Word, N>` | Adds little-endian, extended-width modular `advance(uint64_t)`; it is not a generator. |
| `rng::Philox<N, W, R>` | C++26 `std::philox_engine` | RandBLAS is a stateless `(counter, key) -> block` function. The standard engine owns state, caches a block position, and returns one scalar per mutating `operator()`. |
| `RNGState<Engine>` | State stored inside a standard random-number engine | Exposes nonmutating block generation and explicit block advancement only. It has no scalar `operator()`, serialization, seed sequence, or cached lane index. |
| `rng::RepackedOutput<Engine, Word>` | `std::independent_bits_engine` | Re-expresses every bit of one existing block in fixed LSB-first chunks. It does not draw a variable number of scalar values or define a new stream position. |
| `rng::u01`, `rng::uneg11`, `rng::boxmuller` | `std::uniform_real_distribution` and `std::normal_distribution` | Preserve the current mappings and fixed block consumption. Standard distributions do not promise the required mapping or consumption pattern. |
| `CounterBasedRNGState` | `std::uniform_random_bit_generator` | Produces a fixed result block without mutation; a URBG produces one scalar by mutating itself. Neither native RandBLAS concept models URBG. |

Neither a RandBLAS engine nor state is a standard uniform random bit generator.
A scalar standard-engine adaptor can be designed independently if one is ever
needed.

## Counter and seed semantics

Each engine chooses its `ctr_t`, including the counter's period and the meaning
of `advance(1)`. The counter type implements modular `advance(uint64_t)`;
generic state and sampler code delegates to that operation.

An engine may provide `static key_t make_key(uint64_t)`. Only engines with that
hook support `RNGState(uint64_t)`. Explicit key and counter/key construction
remains available for known-answer tests and expert use. Native Philox preserves
the existing scalar-seed interpretation: start with a zero key and add the
64-bit seed using the key's extended-width unsigned representation.

## Output blocks and repacking order

`res_t` is a fixed-size array of unsigned words. `RepackedOutput` preserves all
bits and the wrapped engine's block boundary while exposing narrower unsigned
lanes. Source-word order is preserved, and chunks within a word are emitted
least-significant first. Thus `0xAABBCCDD` becomes `{0xCCDD, 0xAABB}` for
16-bit output and `{0xDD, 0xCC, 0xBB, 0xAA}` for 8-bit output, independently of
host byte order.

Current samplers continue to consume native 32- or 64-bit lanes. This migration
does not define bit assembly for 8- or 16-bit sampler inputs.

## Coordinate-addressed sampling

Sampling is indexed by coordinates and reserved counter blocks, not by the
execution order of a shared stream. A worker copies the input state, advances
that copy by the block offset for its region, generates local blocks, and leaves
the caller's state unchanged. The returned state is a copy advanced by the
total reservation.

This mapping is what makes full/submatrix generation consistent and makes
generated operators independent of OpenMP thread count. Dense row padding and
sparse block reservations are compatibility constraints during the migration.

## Floating-point reproducibility

Native transforms retain the integer-to-floating formulas, constants,
endpoints, word assignment, and output order used by RandBLAS through
Random123. They use `std::sin`, `std::cos`, `std::log`, and `std::sqrt` on the
host. Dense Gaussian values may therefore differ in the last bits across math
libraries, compilers, and architectures. The integer Philox stream and default
sparse operator output remain bitwise compatibility requirements.

Standard-library distributions are not substituted because their mappings and
engine-consumption patterns are not portable, and some have cached or
variable-consumption behavior.

## Algorithm provenance and licensing

The Philox algorithm, floating-point transformations, and known-answer material
are adapted from D. E. Shaw Research's Random123 project. Files containing
adapted implementation or test material retain the applicable D. E. Shaw
Research BSD-3-Clause notice. Developer documentation will cite the Philox
paper and the exact pinned Random123 revision used to generate static vectors.

Native Philox is a statistical counter-based generator, not a cryptographic
random-number generator.

## Known-answer and statistical testing

Static known-answer vectors cover each supported Philox word count, word width,
and round count. Those vectors are generated once from the pinned Random123
checkout; normal builds and tests never locate Random123. Separate tests cover
counter carries and wraparound, engine/state concepts, seed mapping, output
repacking, floating-point endpoints, Box--Muller reference values, statistical
behavior, sampler state advancement, full/submatrix agreement, and OpenMP
thread-count independence.

Characterization fixtures captured before the migration protect the default
dense and sparse streams. Installed-package and example builds protect the
absence of a transitive Random123 dependency.

## Adding another engine

A new engine supplies value-semantic `ctr_t`, `key_t`, and fixed unsigned
`res_t` types plus the output-only `generate` function. Its counter supplies
`advance(uint64_t)`. It may supply `make_key(uint64_t)` if scalar seeding has a
stable, documented mapping. No inheritance or Philox-specific representation is
required.

The engine should have direct known-answer tests, counter-period tests, seed
mapping tests, and appropriate statistical validation before it is used by a
sampler. Sampler-specific result width and lane-count requirements remain
separate from the base engine concept.

## Performance validation

The migration records pre-change and native timings for the basic dense RNG
benchmark and sparse sketch sampling benchmark under the same compiler, build
type, dimensions, and thread count. A regression outside ordinary run-to-run
variation must be investigated before merge. Optimization beyond parity is not
part of the dependency-removal change.
