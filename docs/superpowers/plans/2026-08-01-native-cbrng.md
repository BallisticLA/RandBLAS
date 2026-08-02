# Native Counter-Based RNG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RandBLAS's Random123 dependency with native, bit-compatible Philox engines, native floating-point transforms, a structural state API, and the `RepackedOutput` adaptor, without changing the default sparse stream or coordinate-addressed sampling behavior.

**Architecture:** Implement small header-only RNG primitives under `RandBLAS/rng/`, expose them through `RandBLAS/random_gen.hh`, and make all sampling code depend on the state-like `generate(res_t&)`/`advance(uint64_t)` boundary. Keep counter arithmetic, scalar seed mapping, and output representation owned by the engine or adaptor. Remove Random123 from source, tests, build metadata, installed packages, examples, CI, and installation documentation only after the native path and characterization tests pass.

**Tech Stack:** C++20, CMake, GoogleTest, OpenMP, BLAS++, Spack-provided LLVM/CMake/GoogleTest, GitHub Actions, PowerShell for Windows CI.

## Global Constraints

- The approved design is [2026-07-31-native-cbrng-design.md](../specs/2026-07-31-native-cbrng-design.md). If this plan and the design disagree, stop and amend the plan before changing code.
- Follow the workspace and repository `AGENTS.md` files. All builds and tests use `/Users/riley/randnla/dev/sourceme.sh`.
- Preserve thread-count-independent, coordinate-addressed sampling. Never replace counter offsets with a shared sequential stream.
- Preserve the exact default integer stream and exact default sparse-sketch output bits. Dense floating-point output may differ only by IEEE-compliant host math-library rounding in `sin`, `cos`, `log`, and `sqrt`.
- Keep this PR header-only. Do not add a compiled RandBLAS RNG library.
- Keep `generate` output-only: `void generate(ctr_t const&, key_t const&, res_t&) const` for engines and `void generate(res_t&) const` for states.
- Use the approved aliases `ctr_t`, `key_t`, and `res_t`; do not introduce the old `counter_type`, `key_type`, or `result_type` spellings.
- Algorithms template on state types, never on an engine plus exposed counter/key values.
- Do not make the native engine or state model `std::uniform_random_bit_generator` in this PR.
- Do not make current samplers consume 8- or 16-bit `RepackedOutput` lanes. Reject unsupported sampler result shapes with clear compile-time diagnostics.
- Directly adapted Random123 algorithms, constants, comments, and test vectors retain the D. E. Shaw Research BSD-3-Clause notice and provenance.
- Tests must not locate or include Random123 after the migration. Offline vector generation may use `/Users/riley/randnla/dev/repo-deps/random123`, but generated vectors must be static checked-in data.
- Preserve GNU, Clang, Apple Clang, and MSVC support. Native headers must remain host-parseable in CUDA-aware/NVCC configurations; no CUDA device API is added.
- Do not change RandLAPACK in this plan.
- Preserve the pre-existing untracked `.claude/` directory and unrelated user changes.

---

## Execution protocol and review checkpoints

Execute tasks in order. For each task:

1. Check the task's repository status and confirm only expected files are dirty.
2. Add the specified test or characterization first.
3. Run the narrow command and observe the expected failure, unless the step is explicitly a passing characterization test or documentation-only step.
4. Make the minimum implementation change.
5. Run the narrow test, then the task-level regression command.
6. Check off completed steps in this file and add the commit hash to the execution log.
7. Commit only that task's files with the listed commit message.

Do not batch past these mid-PR review points unless the reviewer explicitly asks:

- **Checkpoint A — native primitives:** after Task 5. Philox, repacking, and transforms work, while the old sampling path may still use Random123.
- **Checkpoint B — public API migration:** after Task 6. Native state and all source/test/example call sites work, while CMake/CI dependency cleanup may still be pending.
- **Checkpoint C — dependency-free package:** after Task 8. Local and installed builds no longer know about Random123.

Update this table as work lands; record benchmark medians and links to any CI runs in the Notes column.

| Task | Status | Commit | Notes |
|---|---|---|---|
| 1. Characterize behavior and record baseline | Complete | This commit | LLVM/Clang 19.1.3, Release, one thread. Dense 8192x1024 median 16,561,709 ticks; range 16,430,125–31,743,500. Sparse left/ColMajor warm min/median 4,226/4,280 us; COLD min 4,390 us. |
| 2. Add full-width word arrays | Not started | — | — |
| 3. Add native Philox and static KATs | Not started | — | — |
| 4. Add `RepackedOutput` | Not started | — | — |
| 5. Add native floating-point transforms | Not started | — | — |
| 6. Migrate state and sampler APIs atomically | Not started | — | — |
| 7. Remove the build/package dependency | Not started | — | — |
| 8. Remove Random123 from CI | Not started | — | — |
| 9. Finish user and developer documentation | Not started | — | — |
| 10. Run final validation and performance comparison | Not started | — | — |

---

## Standard-library comparison to preserve during implementation

The permanent version of this table belongs in `RandBLAS/rng/DevNotes.md`.

| RandBLAS abstraction | Nearest standard-library abstraction | Deliberate difference |
|---|---|---|
| `rng::WordArray<Word, N>` | `std::array<Word, N>` | Adds little-endian, extended-width modular `advance(uint64_t)`; it is not a generator. |
| `rng::Philox<N, W, R>` | C++26 `std::philox_engine` | RandBLAS is a stateless `(counter, key) -> block` function. The standard engine owns state, caches a block position, and returns one scalar per mutating `operator()`. |
| `RNGState<Engine>` | The state stored inside a standard random-number engine | Exposes nonmutating block generation and explicit block advancement only. It has no scalar `operator()`, serialization, seeding sequence, or cached lane index. |
| `rng::RepackedOutput<Engine, Word>` | `std::independent_bits_engine` | Re-expresses every bit of one existing block in fixed LSB-first chunks. It does not draw variable numbers of scalar values or define a new stream position. |
| `rng::u01`, `rng::uneg11`, `rng::boxmuller` | `std::uniform_real_distribution` and `std::normal_distribution` | Preserve the current Random123 mappings and fixed block consumption. Standard distributions do not promise the required mapping or consumption pattern. |
| `CounterBasedRNGState` | `std::uniform_random_bit_generator` | Produces a fixed result block without mutation; a URBG produces one scalar by mutating itself. Neither native RandBLAS concept models URBG. |

---

### Task 1: Characterize current behavior and record the baseline

**Files:**

- Create: `RandBLAS/rng/DevNotes.md`
- Create: `test/basic_rng/test_sampler_regression.cc`
- Modify: `RandBLAS/DevNotes.md`
- Modify: `test/CMakeLists.txt`
- Modify during execution: `docs/superpowers/plans/2026-08-01-native-cbrng.md` (execution log only)

**Interfaces consumed:** Existing Random123-backed `RNGState<>`, `fill_dense_unpacked`, `fill_sparse_unpacked`, `DenseSkOp`, and `SparseSkOp`.

**Interfaces produced:** A checked-in behavioral oracle for the default stream; a permanent RNG developer-notes entry point; reproducible pre-change benchmark numbers.

- [x] **Step 1: Verify the starting branch and full test suite**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
git status --short --branch
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j
ctest --test-dir build-randblas --output-on-failure
```

Expected: the branch is `native-cbrng`; only known user files are untracked/modified; all existing tests pass before characterization is added.

- [x] **Step 2: Add a deterministic sampler characterization test while Random123 is still active**

Add `test/basic_rng/test_sampler_regression.cc` to `STAT_SOURCES`. Cover these fixed cases with `RNGState<>(0x0123456789abcdefULL)`:

- dense uniform and Gaussian `DenseDist(3, 7)`, including the returned state;
- short-axis sparse `SparseDist(5, 11, 3, Axis::Short)`;
- long-axis sparse `SparseDist(5, 11, 3, Axis::Long)`;
- a one-nonzero sparse case exercising `sample_indices_iid_uniform`.

For sparse cases, compare `rows`, `cols`, and the raw object representation of each `float` value so the test is bitwise, not tolerance-based. Compare dense uniform values bitwise; compare dense Gaussian values with an epsilon-scaled tolerance that permits only the approved host-math rounding boundary. Retain the existing thread-count and submatrix tests for stronger structural invariants. Compare returned states against explicit counter/key values.

Use a temporary, uncommitted printer built against the current code to emit the constants. Inspect its output once, copy the constants into the test, and delete the printer before committing. The checked-in test itself must contain no runtime reference implementation and no path to Random123.

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
ctest --test-dir build-randblas --output-on-failure -R 'SamplerRegression'
```

Expected: the characterization tests pass against the existing implementation.

- [x] **Step 3: Record the API rationale and provenance scaffold**

Create `RandBLAS/rng/DevNotes.md` with these headings and fill each with the decisions from the approved design:

```markdown
# Random-number generation developer notes

## Public engine and state contracts
## Relationship to the C++ standard random facilities
## Counter and seed semantics
## Output blocks and repacking order
## Coordinate-addressed sampling
## Floating-point reproducibility
## Algorithm provenance and licensing
## Known-answer and statistical testing
## Adding another engine
## Performance validation
```

Include the standard-library comparison table above and link this file from `RandBLAS/DevNotes.md`. At this stage, describe the approved target architecture and clearly label Random123 removal as in progress.

- [x] **Step 4: Capture reproducible pre-change performance numbers**

Build and run seven single-thread trials of the direct dense RNG benchmark:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target test_rng_speed
for trial in 1 2 3 4 5 6 7; do
    OMP_NUM_THREADS=1 ./build-randblas/bin/test_rng_speed 8192 1024
done
```

Install the current library, rebuild examples, and record the sparse benchmark's warm and COLD fields for seven internal trials. The current benchmark does not emit a standalone SAMPLE field, so do not infer one by subtracting two noisy timings:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target install
cmake --build build-randblas-examples -j --target sketch_general_performance
OMP_NUM_THREADS=1 ./build-randblas-examples/sketch_general_performance --no-stream 200 2000 2000 4 0 7
```

Record compiler identity, build type, `OMP_NUM_THREADS`, direct benchmark median/range, and sparse warm/COLD output in this plan's execution log. Do not commit raw generated binaries or logs.

- [x] **Step 5: Commit the characterization checkpoint**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
git diff --check
git status --short
git add RandBLAS/rng/DevNotes.md RandBLAS/DevNotes.md test/basic_rng/test_sampler_regression.cc test/CMakeLists.txt docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "test: characterize Random123-backed sampling"
```

---

### Task 2: Add full-width word arrays and modular advancement

**Files:**

- Create: `RandBLAS/rng/word_array.hh`
- Create: `test/basic_rng/test_word_array.cc`
- Modify: `RandBLAS/random_gen.hh`
- Modify: `test/CMakeLists.txt`

**Interfaces consumed:** `std::array`, unsigned modular arithmetic.

**Interfaces produced:** `RandBLAS::rng::WordArray<Word, WordCount>`, used as Philox `ctr_t` and `key_t`.

- [ ] **Step 1: Write failing counter arithmetic and value-semantics tests**

Add `test/basic_rng/test_word_array.cc` to `STAT_SOURCES`. Its core cases must be equivalent to:

```cpp
using A = RandBLAS::rng::WordArray<std::uint32_t, 4>;

TEST(WordArray, AdvancesWithCarryFromLeastSignificantWord) {
    A value{{0xffffffffu, 7u, 9u, 11u}};
    value.advance(2);
    EXPECT_EQ(value, (A{{1u, 8u, 9u, 11u}}));
}

TEST(WordArray, WrapsAtFullWidth) {
    A value{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
    value.advance(1);
    EXPECT_EQ(value, A{});
}
```

Also test zero advance, a carry through multiple words, a `uint64_t` advance into 32-bit words, indexing, `size()`, copy/equality, and `WordArray<uint64_t, 2>` advancement.

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
```

Expected: compilation fails because `RandBLAS/rng/word_array.hh` and `WordArray` do not exist.

- [ ] **Step 2: Implement the minimal full-width value type**

Implement the public shape:

```cpp
namespace RandBLAS::rng {

template <std::unsigned_integral Word, std::size_t WordCount>
struct WordArray {
    using value_type = Word;
    static constexpr std::size_t static_size = WordCount;

    std::array<Word, WordCount> words{};

    constexpr Word& operator[](std::size_t i) noexcept { return words[i]; }
    constexpr Word const& operator[](std::size_t i) const noexcept { return words[i]; }
    [[nodiscard]] static constexpr std::size_t size() noexcept { return WordCount; }
    constexpr void advance(std::uint64_t amount) noexcept;
    friend constexpr bool operator==(WordArray const&, WordArray const&) = default;
};

}
```

`advance` treats word zero as least significant, adds all bits of the 64-bit amount, propagates carry toward higher indices, and discards carry beyond `WordCount`. Avoid signed overflow and byte-order-dependent code. Include this header from `RandBLAS/random_gen.hh` without changing the default engine yet.

- [ ] **Step 3: Verify the focused and statistical suites**

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
ctest --test-dir build-randblas --output-on-failure -R 'WordArray'
ctest --test-dir build-randblas --output-on-failure -R 'SamplerRegression|WordArray'
```

Expected: all listed tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git diff --check
git add RandBLAS/rng/word_array.hh RandBLAS/random_gen.hh test/basic_rng/test_word_array.cc test/CMakeLists.txt docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "feat: add native RNG word arrays"
```

---

### Task 3: Add native Philox and static known-answer tests

**Files:**

- Create: `RandBLAS/rng/philox.hh`
- Create: `test/basic_rng/test_philox.cc`
- Create: `test/basic_rng/philox_kat_vectors.txt`
- Modify: `RandBLAS/random_gen.hh`
- Modify: `test/CMakeLists.txt`
- Delete: `test/basic_rng/test_r123.cc`
- Delete: `test/basic_rng/r123_kat_vectors.txt`
- Delete: `test/basic_rng/r123_rngNxW.mm`

**Interfaces consumed:** `rng::WordArray`; the pinned Random123 checkout only as an offline oracle.

**Interfaces produced:** `RandBLAS::rng::Philox<N, W, R>` for `N in {2,4}`, `W in {32,64}`, and `R in [0,16]`, with aliases `ctr_t`, `key_t`, `res_t`, `generate`, and `make_key`.

- [ ] **Step 1: Generate and check in independent static vectors**

Use `/Users/riley/randnla/dev/repo-deps/random123` outside the RandBLAS build to generate three nontrivial `(counter, key, result)` rows for every one of the 68 engine specializations. Include round zero and rounds 1 through 16. The three inputs per specialization must include:

1. all-zero counter and key;
2. the first published/nonzero input already represented for that family in `r123_kat_vectors.txt`;
3. carry-heavy alternating words (`0xffffffff`/`0xffffffffffffffff`, `1`, and the high bit) to exercise multiplication and word order.

Use a text format with one family, round count, all counter words, all key words, and all result words per line. Copy the existing D. E. Shaw Research BSD-3-Clause notice and add a comment identifying the pinned Random123 commit returned by:

```bash
git -C /Users/riley/randnla/dev/repo-deps/random123 rev-parse HEAD
```

The generator is a temporary offline tool and must not be added to RandBLAS. Confirm the fixture contains `4 families * 17 rounds * 3 inputs = 204` data rows.

- [ ] **Step 2: Replace the inherited Random123 test with failing native tests**

Replace `test_r123.cc` in `STAT_SOURCES` with `test_philox.cc`; change the baked path definition to:

```cmake
target_compile_definitions(stat_tests PRIVATE
    PHILOX_KAT_VECTORS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/basic_rng/philox_kat_vectors.txt")
```

The test parser must dispatch all rounds at compile time, for example with `std::make_index_sequence<17>`, so each row instantiates the exact `Philox<N, W, R>` type. The central check is:

```cpp
typename Engine::res_t actual;
using word_t = typename Engine::res_t::value_type;
actual.fill(std::numeric_limits<word_t>::max());
auto counter_before = counter;
auto key_before = key;

Engine{}.generate(counter, key, actual);

EXPECT_EQ(actual, expected);
EXPECT_EQ(counter, counter_before);
EXPECT_EQ(key, key_before);
```

Also assert:

- `Philox<N,W,0>` copies the counter into the output;
- `res_t` has `N` unsigned `W`-bit words;
- `ctr_t` has `N` words and `key_t` has `N/2` words;
- `make_key(0)` is zero and `make_key(seed)` matches the old zero-key-plus-`incr(seed)` interpretation;
- `generate` overwrites every pre-poisoned output lane.

Delete the Threefry, `MicroURNG`, conventional `Engine`, and unsupported Random123-only tests with the old source and `.mm` file.

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
```

Expected: compilation fails because `rng::Philox` does not exist.

- [ ] **Step 3: Implement portable Philox multiplication and rounds**

Implement the public form:

```cpp
template <std::size_t N, std::size_t W, std::size_t R>
class Philox {
    static_assert(N == 2 || N == 4);
    static_assert(W == 32 || W == 64);
    static_assert(R <= 16);

public:
    using word_t = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
    using ctr_t = WordArray<word_t, N>;
    using key_t = WordArray<word_t, N / 2>;
    using res_t = std::array<word_t, N>;

    static constexpr key_t make_key(std::uint64_t seed) noexcept;
    constexpr void generate(ctr_t const& counter,
                            key_t const& key,
                            res_t& output) const noexcept;
};
```

Match Random123's constants, multiply-high/low operation, round permutation, XORs, and Weyl key bumps exactly. For 32-bit words, multiply in `uint64_t`. For 64-bit words, use `unsigned __int128` on GNU/Clang/Apple Clang and `_umul128` from `<intrin.h>` on 64-bit MSVC. Keep compiler-specific code in a small internal `mulhilo` helper, with compile-time diagnostics for an unsupported 64-bit host path. Do not use signed arithmetic or reinterpret casts for word order.

For each round, transform the block with the current key; bump the key only when another round follows. For `R == 0`, write the input counter words directly. Include the applicable D. E. Shaw Research notice in this adapted header.

- [ ] **Step 4: Run KATs and compile-time API checks**

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
ctest --test-dir build-randblas --output-on-failure -R 'Philox'
ctest --test-dir build-randblas --output-on-failure -R 'SamplerRegression|WordArray|Philox'
```

Expected: all 204 vectors and all API tests pass; the still-Random123-backed sampler characterization remains unchanged.

- [ ] **Step 5: Scan the new tests for accidental dependency leakage and commit**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'Random123/|find_package\(Random123|r123::' test/basic_rng/test_philox.cc test/basic_rng/philox_kat_vectors.txt RandBLAS/rng
git diff --check
```

Expected: only attribution/provenance comments mention Random123; no include, namespace use, or package lookup appears.

```bash
git add RandBLAS/rng/philox.hh RandBLAS/random_gen.hh test/CMakeLists.txt test/basic_rng/test_philox.cc test/basic_rng/philox_kat_vectors.txt test/basic_rng/test_r123.cc test/basic_rng/r123_kat_vectors.txt test/basic_rng/r123_rngNxW.mm docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "feat: add bit-compatible native Philox"
```

---

### Task 4: Add `RepackedOutput`

**Files:**

- Create: `RandBLAS/rng/repacked_output.hh`
- Create: `test/basic_rng/test_repacked_output.cc`
- Modify: `RandBLAS/random_gen.hh`
- Modify: `test/CMakeLists.txt`

**Interfaces consumed:** Any conforming stateless engine's `ctr_t`, `key_t`, `res_t`, `generate`, and optional `make_key`.

**Interfaces produced:** `RandBLAS::rng::RepackedOutput<Engine, OutputWord>`.

- [ ] **Step 1: Write failing direct, nested, forwarding, and rejection tests**

Use a deterministic test engine returning:

```cpp
std::array<std::uint32_t, 2>{0xaabbccddu, 0x01234567u}
```

Assert exact results:

```cpp
EXPECT_EQ(out16, (std::array<std::uint16_t, 4>{
    0xccddu, 0xaabbu, 0x4567u, 0x0123u
}));
EXPECT_EQ(out8, (std::array<std::uint8_t, 8>{
    0xddu, 0xccu, 0xbbu, 0xaau, 0x67u, 0x45u, 0x23u, 0x01u
}));
```

Add tests for:

- direct `Philox<4,32,10> -> uint16_t` and `-> uint8_t`;
- nested `Philox<4,32,10> -> uint16_t -> uint8_t` equality with direct `-> uint8_t`;
- total block bit count;
- exact `ctr_t` and `key_t` identity with the wrapped engine;
- forwarding of `make_key` only when present;
- rejection of signed, wider, non-dividing, and non-power-of-two output word widths with compile-time `requires` assertions.

The state-level repacking test belongs to Task 6, after the final `RNGState` API exists.

Run the `stat_tests` target. Expected: compilation fails because `RepackedOutput` does not exist.

- [ ] **Step 2: Implement shift-and-mask repacking**

Implement:

```cpp
template <class Engine, std::unsigned_integral OutputWord>
    requires detail::EngineHasFixedUnsignedResult<Engine>
          && ValidRepacking<typename Engine::res_t::value_type, OutputWord>
class RepackedOutput {
public:
    using ctr_t = typename Engine::ctr_t;
    using key_t = typename Engine::key_t;
    using res_t = std::array<OutputWord, repacked_word_count>;

    void generate(ctr_t const& counter,
                  key_t const& key,
                  res_t& output) const;
};
```

Generate once into `Engine::res_t`, then emit each source word's chunks from least significant to most significant using unsigned shifts and masks. Preserve source-word order. Do not use object representation, `memcpy`, unions, or host endianness. Forward `make_key(uint64_t)` with a constrained static member when the wrapped engine has it. Store the wrapped engine with `[[no_unique_address]]` so nested adaptors remain cheap.

`ValidRepacking` requires an unsigned output word, no widening, an exact bit-width division, and a power-of-two width ratio. Equal-width adaptation may either be accepted as an identity adaptor or rejected consistently; choose identity because it composes naturally and document/test it.

- [ ] **Step 3: Verify repacking and native KAT regressions**

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
ctest --test-dir build-randblas --output-on-failure -R 'RepackedOutput|Philox'
```

Expected: direct/nested outputs and compile-time contract checks pass; Philox KATs remain green.

- [ ] **Step 4: Commit Checkpoint A's engine-adaptor portion**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git diff --check
git add RandBLAS/rng/repacked_output.hh RandBLAS/random_gen.hh test/basic_rng/test_repacked_output.cc test/CMakeLists.txt docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "feat: add block output repacking"
```

---

### Task 5: Add native integer-to-floating transforms

**Files:**

- Create: `RandBLAS/rng/distributions.hh`
- Create: `test/basic_rng/test_distributions.cc`
- Modify: `RandBLAS/random_gen.hh`
- Modify: `test/CMakeLists.txt`

**Interfaces consumed:** Fixed-size arrays of unsigned 32- or 64-bit words.

**Interfaces produced:** Native `u01`, `uneg11`, block conversion, Box--Muller, and dense transform policies under `RandBLAS::rng`.

- [ ] **Step 1: Write failing endpoint and reference tests**

Adapt only the Random123 conversion and Box--Muller cases RandBLAS actually uses. Test `uint32_t -> float`, `uint32_t -> double` where used, and `uint64_t -> double`. Include zero, one, midpoint/high-bit, maximum, and the reference values retained from the old test. Verify endpoint openness/closedness explicitly.

For block conversion, assert output length and per-lane correspondence. For Box--Muller, use fixed integer pairs and compare both outputs with a tolerance based on `std::numeric_limits<T>::epsilon()` and the result magnitude. Verify which word supplies angle/radius and which returned lane is sine/cosine by using asymmetric inputs.

The policy-level test should have this shape:

```cpp
typename State::res_t bits{};
state.generate(bits);
auto uniform = RandBLAS::rng::uneg11::generate(state);
auto normal = RandBLAS::rng::boxmul::generate(state);
EXPECT_EQ(uniform.size(), bits.size());
EXPECT_EQ(normal.size(), bits.size());
```

Until the final native `RNGState` lands in Task 6, use a minimal test-only state satisfying `generate(res_t&) const`.

Run `stat_tests`. Expected: compilation fails because the native transform header and functions do not exist.

- [ ] **Step 2: Implement the retained formulas faithfully**

Implement scalar and block helpers using the same constants, scaling, endpoint convention, precision selection, angle/radius assignment, and sine/cosine output order as the current Random123-backed code. Preserve the rule that 32-bit source words produce `float` by default and 64-bit words produce `double` by default. The Box--Muller block length must be even.

Expose structurally generic policy wrappers usable by dense sampling:

```cpp
struct uneg11 {
    template <class State>
        requires detail::StateCanGenerateFixedUnsignedBlock<State>
    static auto generate(State const& state);
};

struct boxmul {
    template <class State>
        requires detail::StateCanGenerateFixedUnsignedBlock<State>
    static auto generate(State const& state);
};
```

`detail::StateCanGenerateFixedUnsignedBlock` here is a local structural requirement in the distribution header; it must not depend on the umbrella header or create an include cycle. Task 6's public `CounterBasedRNGState` concept is the authoritative sampler boundary and must accept the same test state. Each wrapper fills a local `State::res_t`, calls `state.generate`, and applies the pure transform. It does not advance the state. Use `std::sin`, `std::cos`, `std::log`, and `std::sqrt`; remove the global `sincospi` shim only in Task 6 when Random123 headers are removed. Retain applicable D. E. Shaw Research notices.

- [ ] **Step 3: Verify native transforms and statistical tests**

Run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests
ctest --test-dir build-randblas --output-on-failure -R 'Distribution|Continuous|Distortion|SamplerRegression'
```

Expected: native reference tests pass, and existing Random123-backed statistical/characterization tests remain green.

- [ ] **Step 4: Commit Checkpoint A**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git diff --check
git add RandBLAS/rng/distributions.hh RandBLAS/random_gen.hh test/basic_rng/test_distributions.cc test/CMakeLists.txt docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "feat: add native random transforms"
```

Pause for Checkpoint A review if requested.

---

### Task 6: Migrate state and sampler APIs atomically

**Files:**

- Modify: `RandBLAS/random_gen.hh`
- Modify: `RandBLAS/base.hh`
- Modify: `RandBLAS/dense_skops.hh`
- Modify: `RandBLAS/sparse_skops.hh`
- Modify: `RandBLAS/skge.hh`
- Modify: `RandBLAS/sparse_data/sksp.hh` (template documentation and any state-type spellings)
- Modify: `RandBLAS/util.hh`
- Modify: `RandBLAS/testing/lapack_like.hh`
- Modify: `RandBLAS/testing/linops.hh`
- Modify: `RandBLAS/testing/sparse_data.hh`
- Create: `test/basic_rng/test_rng_state.cc`
- Modify: `test/basic_rng/test_discrete.cc`
- Modify: `test/basic_rng/test_distortion.cc`
- Modify: `test/basic_rng/benchmark_speed.cc`
- Modify: `test/datastructures/test_denseskop.cc`
- Modify: `test/datastructures/test_sparseskop.cc`
- Modify: `test/datastructures/test_coo_matrix.cc`
- Modify: `test/linops/test_lskges.cc`
- Modify: `test/linops/test_rskges.cc`
- Modify: `test/meta/test_sparse_data_generators.cc`
- Modify: `test/test_io.cc`
- Modify: `examples/sparse-low-rank-approx/qrcp_matrixmarket.cc`
- Modify: `examples/sparse-low-rank-approx/svd_matrixmarket.cc`
- Modify: `examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc`
- Modify: `test/CMakeLists.txt`

**Interfaces consumed:** Native `Philox`, `RepackedOutput`, transforms, and `WordArray`.

**Interfaces produced:** `RandBLAS::rng::CounterBasedEngine`, `RandBLAS::CounterBasedRNGState`, `RNGState<Engine>`, `DefaultRNG`, `DefaultRNGState`, state-templated samplers and sketching operators.

- [ ] **Step 1: Write failing structural engine/state tests**

Add `test_rng_state.cc` to `STAT_SOURCES`. Define a test-only engine whose counter's representation is private and unrelated to `WordArray`:

```cpp
class OpaqueCounter {
public:
    constexpr void advance(std::uint64_t blocks) noexcept;
    friend constexpr bool operator==(OpaqueCounter const&, OpaqueCounter const&) = default;
private:
    std::uint64_t value_ = 0;
    friend struct OpaqueEngine;
};

struct OpaqueEngine {
    using ctr_t = OpaqueCounter;
    using key_t = std::array<std::uint32_t, 1>;
    using res_t = std::array<std::uint32_t, 2>;
    static constexpr key_t make_key(std::uint64_t seed) noexcept;
    constexpr void generate(ctr_t const&, key_t const&, res_t&) const noexcept;
};
```

Test:

- `static_assert(rng::CounterBasedEngine<OpaqueEngine>)`;
- `static_assert(CounterBasedRNGState<RNGState<OpaqueEngine>>)`;
- `static_assert(!std::uniform_random_bit_generator<OpaqueEngine>)` and the same for the state;
- default, scalar-seed, explicit-key, and explicit-counter/key construction;
- absence of scalar-seed construction for an engine without `make_key`;
- Rule-of-Zero copy/move/assignment and equality;
- `generate` nonmutation and `advance` delegation;
- const `counter()` and `key()` observation;
- `RNGState<rng::RepackedOutput<DefaultRNG, uint16_t>>` generation and identical block advancement.

Run `stat_tests`. Expected: compilation fails because the concepts and final state API do not exist.

- [ ] **Step 2: Inventory every old representation dependency immediately before editing**

Run and save the output in the task notes:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'r123::|r123ext::|Random123/|ctr_type|key_type|counter\.incr|key\.incr|\.counter\b|\.key\b' RandBLAS test examples --glob '*.{hh,cc}'
```

Every functional match must be migrated in this task or be one of the already-deleted legacy test files. Do not hide a match with a compatibility namespace.

- [ ] **Step 3: Implement concepts, state, and default aliases in the umbrella**

Replace Random123 includes and `r123ext` definitions in `RandBLAS/random_gen.hh` with native includes and structural concepts. Define the engine concept in `RandBLAS::rng` and the state concept in `RandBLAS`; keep any low-level header constraints structurally equivalent without introducing an umbrella-header include cycle. The public state shape is:

```cpp
using DefaultRNG = rng::Philox<4, 32, 10>;

template <rng::CounterBasedEngine Engine = DefaultRNG>
class RNGState {
public:
    using engine_t = Engine;
    using ctr_t = typename Engine::ctr_t;
    using key_t = typename Engine::key_t;
    using res_t = typename Engine::res_t;

    constexpr RNGState() = default;
    explicit constexpr RNGState(std::uint64_t seed)
        requires rng::SeedMappableEngine<Engine>;
    explicit constexpr RNGState(key_t const& key);
    constexpr RNGState(ctr_t const& counter, key_t const& key);

    constexpr void generate(res_t& output) const;
    constexpr void advance(std::uint64_t blocks);
    [[nodiscard]] constexpr ctr_t const& counter() const noexcept;
    [[nodiscard]] constexpr key_t const& key() const noexcept;
    friend constexpr bool operator==(RNGState const&, RNGState const&);

private:
    ctr_t counter_{};
    key_t key_{};
    [[no_unique_address]] Engine engine_{};
};

using DefaultRNGState = RNGState<DefaultRNG>;
```

The engine concept must check copy/value semantics, unsigned fixed-extent `res_t`, counter advancement, and the exact output-only call. The state concept must require copyability, unsigned fixed-extent `res_t`, nonmutating `generate`, and mutating `advance`, without requiring counter/key access. Keep `RNGState<>` as the default spelling. Equality compares counter and key only, so a stateless engine need not add meaningless equality state. Move the old state definition and its manual destructor/copy/memcpy implementation out of `base.hh`; retain stream output using only const accessors.

- [ ] **Step 4: Migrate dense sampling without changing block addresses**

Change `DenseSkOp<T, RNG>` to `DenseSkOp<T, State = DefaultRNGState>` and store `State` directly. Change `DenseDist::sample`, `fill_dense_submat_impl`, `compute_next_state`, `fill_dense_unpacked`, and `fill_dense` similarly. Propagate the state template through dense/sparse overloads in `RandBLAS/skge.hh` without adding engine assumptions there.

The core generation pattern must be:

```cpp
State row_state = seed;
row_state.advance(block_offset);
auto values = Transform::generate(row_state);
row_state.advance(1);
```

Use `std::tuple_size_v<typename State::res_t>` for block length. Preserve current row padding, `ptr_padded`, first/last block boundaries, inter-row stride, OpenMP `schedule(static)`, and total state increment exactly. Compute the return value by copying `seed` and calling `advance(total_blocks)`; do not reconstruct it from exposed counter/key values.

Dispatch `ScalarDist::Gaussian` through `rng::boxmul` and uniform through `rng::uneg11`. Add compile-time diagnostics that dense sampling requires an even result length and 32- or 64-bit result words.

- [ ] **Step 5: Migrate index and sparse sampling without changing default consumption**

In `util.hh`, replace destructuring and raw generator calls with a copied state:

```cpp
state_t work = state;
typename state_t::res_t bits{};
work.generate(bits);
work.advance(1);
```

For `sample_indices_iid`, consume all lanes of each block before advancing to the next. For `sample_indices_iid_uniform`, preserve the default 4x32 interpretation exactly: combine lanes 0 and 1 into the index word and use lane 2's low bit for the Rademacher. State any other supported native result-shape rules explicitly with `if constexpr` and `static_assert`; do not silently draw an extra block.

In `sparse_skops.hh`, change `SparseSkOp<T, RNG, sint_t>` to `SparseSkOp<T, State = DefaultRNGState, sint_t>` and update `SparseDist::sample`, `compute_next_state`, `fill_sparse_unpacked`, helpers, and state members to use only `generate`/`advance`. Propagate that state parameter through `RandBLAS/skge.hh` and relevant `RandBLAS/sparse_data/sksp.hh` declarations/documentation. Preserve the default reservation of one 4x32 block per nonzero and all submatrix skip arithmetic.

- [ ] **Step 6: Migrate testing helpers, tests, benchmark, and examples**

Use composition in `RandBLAS/testing/sparse_data.hh` instead of inheriting from `RNGState`. Its scalar stream owns a `State`, a `State::res_t` buffer, and a lane index; it refills with `state.generate(buffer)` followed by `state.advance(1)`. Replace `r123::u01` and `r123::boxmuller` with native transforms.

Change helper defaults and explicit template arguments from engine types to state types in the listed headers/tests/examples. Mechanical mappings include:

```cpp
r123::Philox4x32              -> RandBLAS::DefaultRNG
RNGState<r123::Philox4x32>    -> RandBLAS::DefaultRNGState
r123ext::uneg11               -> RandBLAS::rng::uneg11
r123ext::boxmul               -> RandBLAS::rng::boxmul
state.counter.incr(amount)    -> state.advance(amount)
state.counter                 -> state.counter()
state.key                     -> state.key()
RNG::ctr_type::static_size    -> std::tuple_size_v<typename State::res_t>
```

Do not apply the first mapping inside an algorithm template: public algorithms take `State`, not `DefaultRNG` or `Engine`.

- [ ] **Step 7: Observe the structural failure, then build all local test executables**

After adding the tests but before production changes, record the expected compile failure. After Steps 3–6, run:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j --target stat_tests densedata_tests sparsedata_tests meta_tests misc_tests test_rng_speed
ctest --test-dir build-randblas --output-on-failure -R 'RNGState|Philox|RepackedOutput|Distribution|SamplerRegression'
ctest --test-dir build-randblas --output-on-failure
```

Expected: all tests pass. In particular, sparse characterization is bitwise unchanged, dense characterization passes, state-advance tests pass, and thread-count/full-submatrix tests pass.

- [ ] **Step 8: Prove source and tests no longer functionally use Random123**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'Random123/|r123::|r123ext::|ctr_type|key_type|counter\.incr|key\.incr' RandBLAS test examples --glob '*.{hh,cc}'
```

Expected: no functional matches. Attribution comments may mention the name `Random123` but must not contain includes, namespaces, old aliases, or calls.

- [ ] **Step 9: Commit Checkpoint B**

```bash
git diff --check
git add RandBLAS test examples docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "refactor: migrate sampling to native RNG states"
```

Pause for Checkpoint B review if requested.

---

### Task 7: Remove Random123 from local builds and installed packages

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `RandBLAS/CMakeLists.txt`
- Modify: `CMake/rb_config.cmake`
- Modify: `CMake/RandBLASConfig.cmake.in`
- Modify: `examples/CMakeLists.txt`
- Delete: `CMake/FindRandom123.cmake`
- Verify: `test/downstream/CMakeLists.txt`
- Verify: `test/downstream/main.cc`

**Interfaces consumed:** Native headers and BLAS++/OpenMP package dependencies.

**Interfaces produced:** A build tree, installed package, downstream consumer, and examples with no Random123 installation or CMake variable.

- [ ] **Step 1: Add a dependency-free package assertion**

Extend the installed downstream smoke test so `test/downstream/main.cc` constructs `DefaultRNGState`, generates a block, advances once, and calls one public dense sampling function. The consumer CMake command must not receive `Random123_DIR` or add a Random123 module path.

Install the current package and configure the downstream consumer with `-DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON`. Expected before CMake cleanup: configuration fails in `RandBLASConfig.cmake` at `find_dependency(Random123)`, even though Random123 is installed elsewhere on the machine. After cleanup, the same option must be harmless and configuration must pass.

- [ ] **Step 2: Remove source-tree and interface dependency declarations**

Make these exact removals:

- remove `find_package(Random123 REQUIRED)` from top-level `CMakeLists.txt`;
- remove `Random123::Random123` from `RandBLAS_libs`;
- remove the `R123_NO_SINCOS` interface definition and Random123-specific MSVC comments;
- retain `/EHsc` and `/Zc:__cplusplus` where still required by RandBLAS itself;
- remove every `${Random123_DIR}` include from `examples/CMakeLists.txt`;
- delete `CMake/FindRandom123.cmake`.

- [ ] **Step 3: Remove the installed transitive dependency**

In `CMake/rb_config.cmake`, remove conversion/storage of `Random123_DIR` and installation of `FindRandom123.cmake`. In `CMake/RandBLASConfig.cmake.in`, remove `Random123_DIR` fallback and `find_dependency(Random123)` while leaving BLAS++, OpenMP, MKL, and version metadata intact.

- [ ] **Step 4: Reconfigure and build with the dependency path explicitly absent**

First find the current cache entries, then create a clean temporary build so an old include directory cannot mask a dependency:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
rg -n 'Random123' build-randblas/CMakeCache.txt
native_build=$(mktemp -d /private/tmp/randblas-native-cbrng-build.XXXXXX)
native_install=$(mktemp -d /private/tmp/randblas-native-cbrng-install.XXXXXX)
cmake -S repo-randblas -B "$native_build" -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX="$native_install" -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON
cmake --build "$native_build" -j
ctest --test-dir "$native_build" --output-on-failure
```

Expected: configure, build, and tests succeed without passing a Random123 location. Run Steps 4–6 in one shell, or record the concrete `native_build` and `native_install` paths in the execution log and restore those two variables when resuming.

- [ ] **Step 5: Install and test the downstream consumer and examples**

Use the clean build's install target, a clean downstream build, and a clean examples build:

```bash
cmake --build "$native_build" -j --target install
downstream_build=$(mktemp -d /private/tmp/randblas-native-cbrng-downstream.XXXXXX)
cmake -S /Users/riley/randnla/dev/repo-randblas/test/downstream -B "$downstream_build" -DCMAKE_PREFIX_PATH="$native_install" -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON
cmake --build "$downstream_build" -j
examples_build=$(mktemp -d /private/tmp/randblas-native-cbrng-examples.XXXXXX)
cmake -S /Users/riley/randnla/dev/repo-randblas/examples -B "$examples_build" -DCMAKE_PREFIX_PATH="$native_install" -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON -DFETCHCONTENT_SOURCE_DIR_FAST_MATRIX_MARKET=/Users/riley/randnla/dev/build-randblas-examples/_deps/fast_matrix_market-src
cmake --build "$examples_build" -j
```

Expected: both consumers configure and compile without `Random123_DIR`.

- [ ] **Step 6: Scan CMake and installed metadata and commit**

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'Random123|R123_' CMakeLists.txt RandBLAS/CMakeLists.txt CMake examples/CMakeLists.txt test/downstream
rg -n 'Random123|R123_' "$native_install"
git diff --check
git add CMakeLists.txt RandBLAS/CMakeLists.txt CMake examples/CMakeLists.txt test/downstream docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "build: remove Random123 package dependency"
```

Expected: no functional build/package match; installed headers may mention Random123 only in license/provenance comments.

---

### Task 8: Remove Random123 from CI dependency setup

**Files:**

- Modify: `.github/actions/setup-randblas-deps/action.yml`
- Modify: `.github/actions/setup-randblas-deps-windows/action.yml`
- Modify: `.github/actions/setup-randblas-deps-windows/setup.ps1`
- Modify: `.github/scripts/windows/run-ci.ps1`
- Modify: `.github/workflows/core.yml`
- Modify: `.github/workflows/downstream-consumer.yml`
- Modify: `.github/workflows/examples.yml`
- Modify: `.github/workflows/thread-sanitizer.yml`

**Interfaces consumed:** Existing CI dependency actions and CMake entry points.

**Interfaces produced:** Unix and Windows CI configurations with no Random123 checkout, cache, input, output, environment variable, or CMake argument.

- [ ] **Step 1: Remove Unix dependency setup and workflow plumbing**

Delete the Random123 clone/install/export steps and any action descriptions that promise it from `.github/actions/setup-randblas-deps/action.yml`. Remove `-DRandom123_DIR=...`, cache keys/paths, and action outputs from the Unix workflows. Keep BLAS++, LAPACK++, GTest, OpenMP, CUDA-aware host, sanitizer, examples, and downstream coverage unchanged.

- [ ] **Step 2: Remove Windows dependency setup and workflow plumbing**

Delete Random123 inputs/cache declarations from the Windows composite action, clone/install/result handling from `setup.ps1`, and required environment/CMake arguments from `run-ci.ps1`. Preserve PowerShell error handling, vcpkg/toolchain behavior, runtime DLL staging, and `/openmp:experimental` behavior.

- [ ] **Step 3: Validate YAML/PowerShell text and local equivalents**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'Random123|R123_|random123' .github
git diff --check
cd /Users/riley/randnla/dev
source sourceme.sh
cmake --build build-randblas -j
ctest --test-dir build-randblas --output-on-failure
```

Expected: no CI matches and the local equivalent remains green. If `actionlint` is already installed, also run `actionlint`; do not add a new tool dependency solely for this task.

- [ ] **Step 4: Commit Checkpoint C**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git add .github docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "ci: stop provisioning Random123"
```

Pause for Checkpoint C review if requested.

---

### Task 9: Finish user and developer documentation

**Files:**

- Modify: `INSTALL.md`
- Modify: `RandBLAS/rng/DevNotes.md`
- Modify: `RandBLAS/DevNotes.md`
- Modify: `test/DevNotes.md`
- Modify: `rtd/source/FAQ.rst`
- Modify: `rtd/source/api_reference/skops_and_dists.rst`
- Modify: `rtd/source/installation/index.rst`
- Modify: `rtd/source/tutorial/distributions.rst`
- Modify: `rtd/source/tutorial/index.rst`
- Modify: `rtd/source/tutorial/sampling_skops.rst`
- Modify: `rtd/source/tutorial/sketch_updates.rst`
- Modify: `rtd/source/updates/index.rst`

**Interfaces consumed:** Final native API and verified behavior.

**Interfaces produced:** Current installation/API/tutorial documentation and complete permanent RNG developer notes.

- [ ] **Step 1: Remove obsolete installation directions**

Delete Random123 from dependency tables, manual install steps, Windows setup, CMake examples, and troubleshooting in `INSTALL.md` and `rtd/source/installation/index.rst`. State that the RNG is header-only and included with RandBLAS; do not make users configure an RNG package path.

- [ ] **Step 2: Update public API and tutorial spellings**

Replace old engine-template examples with state-template examples. Document:

```cpp
using Engine = RandBLAS::rng::Philox<4, 32, 10>;
using State = RandBLAS::RNGState<Engine>;
State state{1234};
Engine::res_t block{};
state.generate(block);
state.advance(1);
```

Also document `DefaultRNGState`, output-only generation, `ctr_t`/`key_t`/`res_t`, const raw accessors, seed mapping, `RepackedOutput` ordering, thread independence, exact Philox integer compatibility, dense math-library reproducibility limits, and non-cryptographic status. Do not imply current samplers accept repacked 8-/16-bit outputs.

- [ ] **Step 3: Finalize developer notes and test notes**

Remove the “in progress” language from `RandBLAS/rng/DevNotes.md`. Include:

- the standard-library comparison table in this plan;
- exact block and counter semantics;
- scalar seed ownership via `make_key`;
- the default stream guarantee;
- output repacking examples `0xAABBCCDD -> {0xCCDD,0xAABB}` and `->{0xDD,0xCC,0xBB,0xAA}`;
- sampler-specific shape constraints;
- algorithm/paper/vector provenance and BSD notices;
- how to add an engine by satisfying concepts rather than inheriting;
- the KAT, statistical, characterization, package, and performance validation strategy.

Update `test/DevNotes.md` to describe `test_philox.cc`, static offline vectors, `test_repacked_output.cc`, `test_rng_state.cc`, transform tests, and sampler characterization. Historical Random123 mentions are allowed only when they explain provenance or migration.

- [ ] **Step 4: Scan documentation and commit**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'Random123|r123::|r123ext::|ctr_type|key_type|Random123_DIR' INSTALL.md RandBLAS rtd test/DevNotes.md
git diff --check
```

Inspect every match. Expected remaining `Random123` matches are attribution, exact-stream compatibility, or migration history only; no installation/API instructions use it. Old namespace/type/CMake spellings have no matches.

```bash
git add INSTALL.md RandBLAS/DevNotes.md RandBLAS/rng/DevNotes.md test/DevNotes.md rtd docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "docs: document native counter-based RNGs"
```

---

### Task 10: Run final validation and performance comparison

**Files:**

- Modify if a defect is found: the smallest responsible implementation/test/documentation file
- Modify: `docs/superpowers/plans/2026-08-01-native-cbrng.md` (final execution log and benchmark results)

**Interfaces consumed:** Entire source tree, installed package, examples, and execution log.

**Interfaces produced:** Evidence that all acceptance criteria hold and a final reviewable plan record.

- [ ] **Step 1: Re-run the complete clean local build and test suite**

Use a new temporary build and install prefix so cached Random123 paths cannot participate. Run Steps 1–3 in one shell, or record the concrete temporary paths in the execution log and restore the variables when resuming:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
final_build=$(mktemp -d /private/tmp/randblas-native-cbrng-final.XXXXXX)
final_install=$(mktemp -d /private/tmp/randblas-native-cbrng-install.XXXXXX)
cmake -S repo-randblas -B "$final_build" -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX="$final_install" -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON
cmake --build "$final_build" -j
ctest --test-dir "$final_build" --output-on-failure
cmake --build "$final_build" -j --target install
```

Expected: clean configure/build/install and all tests pass.

- [ ] **Step 2: Re-run downstream and examples from the clean install**

```bash
downstream_final=$(mktemp -d /private/tmp/randblas-native-cbrng-downstream.XXXXXX)
cmake -S /Users/riley/randnla/dev/repo-randblas/test/downstream -B "$downstream_final" -DCMAKE_PREFIX_PATH="$final_install" -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON
cmake --build "$downstream_final" -j
examples_final=$(mktemp -d /private/tmp/randblas-native-cbrng-examples.XXXXXX)
cmake -S /Users/riley/randnla/dev/repo-randblas/examples -B "$examples_final" -DCMAKE_PREFIX_PATH="$final_install" -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON -DFETCHCONTENT_SOURCE_DIR_FAST_MATRIX_MARKET=/Users/riley/randnla/dev/build-randblas-examples/_deps/fast_matrix_market-src
cmake --build "$examples_final" -j
```

Expected: downstream and all examples compile with no Random123 variable or installation.

- [ ] **Step 3: Re-run matching performance measurements**

Use the same compiler, build type, dimensions, thread count, and trial counts recorded in Task 1:

```bash
for trial in 1 2 3 4 5 6 7; do
    OMP_NUM_THREADS=1 "$final_build/bin/test_rng_speed" 8192 1024
done
OMP_NUM_THREADS=1 "$examples_final/sketch_general_performance" --no-stream 200 2000 2000 4 0 7
```

Record native median/range and sparse warm/COLD fields beside the baseline. Treat a shift outside ordinary baseline run-to-run variation as a failure to investigate, not as an accepted consequence. Any optimization beyond parity needs its own test and before/after evidence.

- [ ] **Step 4: Perform the final dependency, placeholder, and type-consistency scans**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'Random123/|r123::|r123ext::|Random123_DIR|find_package\(Random123|Random123::Random123|R123_' . --glob '!docs/superpowers/specs/**' --glob '!docs/superpowers/plans/**'
rg -n 'counter_type|key_type|result_type|ctr_type|key_type' RandBLAS test examples rtd
rg -n 'TODO|FIXME|XXX|placeholder|not implemented' RandBLAS/rng test/basic_rng RandBLAS/random_gen.hh
git diff --check
git status --short --branch
```

Expected:

- no functional dependency/API/build match;
- no forbidden old alias spelling introduced by this work;
- no placeholders in the implementation or tests;
- remaining Random123 mentions are reviewed BSD attribution/provenance/history only;
- only the plan log or an explicitly understood user file is dirty.

- [ ] **Step 5: Review acceptance criteria one by one**

Cross-check all 14 acceptance criteria in the approved design. In particular, verify the 204 KAT row count, opaque-counter state test, direct/nested repacking tests, bitwise sparse characterization, dense tolerance boundary, thread tests, package consumer, examples, and benchmark comparison. If any criterion lacks direct evidence, add the smallest test or documentation change and rerun its owning suite.

- [ ] **Step 6: Request code review, address findings, and make the final plan-record commit**

Use `superpowers:requesting-code-review` against the full branch diff. After findings are resolved and verification is rerun:

```bash
cd /Users/riley/randnla/dev/repo-randblas
git add docs/superpowers/plans/2026-08-01-native-cbrng.md
git commit -m "docs: record native CBRNG validation"
```

Do not claim completion or push until the final verification output and CI results are available.
