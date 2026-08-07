# Random-number generation developer notes

RandBLAS provides native, header-only counter-based random-number generation.
This document records its public contracts, reproducibility guarantees, and
validation requirements.

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

RandBLAS algorithms consume state-like objects instead of engines directly.
The structural `GeneratorState` concept requires a fixed-size unsigned `res_t`
and these operations:

```cpp
void generate(res_t& output) const;
void advance(std::uint64_t blocks);
```

`RNGState<Engine>` is the provided transparent adapter. Its public `counter`,
`key`, and `engine` values represent the complete state. Generic algorithms
depend only on the `GeneratorState` operations and do not require or inspect
those public members.

## Relationship to the C++ standard random facilities

| RandBLAS abstraction | Nearest standard-library abstraction | Deliberate difference |
|---|---|---|
| `rng::WordArray<Word, N>` | `std::array<Word, N>` | Adds little-endian, extended-width modular `advance(uint64_t)`; it is not a generator. |
| `rng::Philox<N, W, R>` | C++26 `std::philox_engine` | RandBLAS is a stateless `(counter, key) -> block` function. The standard engine owns state, caches a block position, and returns one scalar per mutating `operator()`. |
| `RNGState<Engine>` | State stored inside a standard random-number engine | Provides transparent counter, key, and engine values together with nonmutating block generation and explicit block advancement. It has no scalar `operator()`, serialization, seed sequence, or cached lane index. |
| `rng::RepackedOutput<Engine, Word>` | `std::independent_bits_engine` | Re-expresses every bit of one existing block in fixed LSB-first chunks. It does not draw a variable number of scalar values or define a new stream position. |
| `rng::u01`, `rng::boxmuller`, `rng::uneg11`, `rng::boxmul` | `std::uniform_real_distribution` and `std::normal_distribution` | Transforms explicit words or result blocks without owning or mutating generator state. |
| `GeneratorState` | `std::uniform_random_bit_generator` | Produces a fixed result block without mutation; a URBG produces one scalar by mutating itself. Neither native RandBLAS concept models URBG. |

Neither a RandBLAS engine nor state is a standard uniform random bit generator.
A scalar standard-engine adaptor can be designed independently if one is ever
needed.

## Counter and seed semantics

Each engine chooses its `ctr_t`, including the counter's period and the meaning
of `advance(1)`. The counter type implements modular `advance(uint64_t)`;
generic state and sampler code delegates to that operation.

Native Philox uses `WordArray<word_t, N>` for `ctr_t`. Lane zero is the least
significant word of an `N * W`-bit unsigned integer, and `advance(k)` adds `k`
to that integer modulo `2^(N * W)`. One call to `generate` produces exactly one
`N`-word block at the current counter without mutating the counter or key.

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

`RandBLAS::testing::detail::RNGStream` is test infrastructure only. It adapts
result blocks to sequential scalar draws for random sparse test-matrix
generation. See `test/DevNotes.md` for its consumption details.

## Floating-point reproducibility

Native transforms retain the integer-to-floating formulas, constants,
endpoints, word assignment, and output order used by RandBLAS through
Random123. They use `std::sin`, `std::cos`, `std::log`, and `std::sqrt` on the
host. Dense Gaussian values may therefore differ in the last bits across math
libraries, compilers, and architectures. The integer Philox stream and default
sparse operator output remain bitwise compatibility requirements.

The supported transform names are `u01`, `boxmuller`, `uneg11`, and `boxmul`.

## Algorithm provenance and licensing

The Philox algorithm is described by Salmon, Moraes, Dror, and Shaw in
[Parallel Random Numbers: As Easy as 1, 2, 3](https://doi.org/10.1145/2063384.2063405).
The implementation, floating-point transformations, and known-answer material
are adapted from D. E. Shaw Research's Random123 project. Files containing
adapted implementation or test material retain the applicable D. E. Shaw
Research BSD-3-Clause notice. Static vectors were generated from Random123
commit [`9545ff6413f258be2f04c1d319d99aaef7521150`](https://github.com/DEShawResearch/random123/commit/9545ff6413f258be2f04c1d319d99aaef7521150).

Native Philox is a statistical counter-based generator, not a cryptographic
random-number generator.

The default engine is `rng::Philox<4, 32, 10>`. Its integer blocks are bitwise
identical to Random123 Philox4x32-10 for the same counter and key. This guarantee
also covers the default sparse sketch stream. Dense Gaussian sampling preserves
the same formulas and block assignment, subject to the host-math limitation
described above.

## Known-answer and statistical testing

Static known-answer vectors cover each supported Philox word count, word width,
and round count. Those vectors are generated once from the pinned Random123
commit `9545ff6413f258be2f04c1d319d99aaef7521150`; normal builds and tests never
locate Random123. Separate tests cover
counter carries and wraparound, engine/state concepts, seed mapping, output
repacking, floating-point endpoints, Box--Muller reference values, statistical
behavior, sampler state advancement, full/submatrix agreement, and OpenMP
thread-count independence.

Characterization fixtures captured before the migration protect the default
dense and sparse streams. Installed-package and example builds protect the
absence of a transitive Random123 dependency.

Dense samplers require 32- or 64-bit result words and an even number of lanes.
`sample_indices_iid_uniform` requires at least two 32-bit lanes (three when
also producing Rademacher signs), or at least one 64-bit lane (two with signs).
Sparse Fisher--Yates sampling requires at least three 32-bit lanes or two
64-bit lanes. These constraints are sampler contracts, not requirements of the
base engine and state concepts. In particular, current samplers do not accept
8- or 16-bit `RepackedOutput` results.

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
