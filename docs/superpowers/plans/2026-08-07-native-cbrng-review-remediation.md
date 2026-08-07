# Native CBRNG Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every point in the second review of RandBLAS PR 182 while preserving the native Philox stream, sampler behavior, and dependency-free package.

**Architecture:** Keep `RNGState<Engine>` as RandBLAS's concrete engine-to-state adapter and rename its structural customization concept to `GeneratorState`. Extract the engine/state concepts into a dependency-light header so distribution policies can use `GeneratorState` without an include cycle. Make the three new RNG data types transparent structs, simplify floating-point transforms around their scalar formulas, and move the sequential scalar stream into focused test-only infrastructure.

**Tech Stack:** C++20 concepts and templates, header-only RandBLAS, GoogleTest, CMake, Spack-provided compiler/dependencies, Sphinx/Doxygen, GitHub CLI.

## Global Constraints

- Follow `/Users/riley/randnla/dev/AGENTS.md` and `/Users/riley/randnla/dev/repo-randblas/AGENTS.md`.
- Run RandBLAS builds and tests from `/Users/riley/randnla/dev/build-randblas` after `source sourceme.sh`.
- Preserve thread-count-independent, coordinate-addressed sampling and all existing state-advance rules.
- Preserve every Philox known-answer vector, the exact default integer stream, and bitwise default sparse-sketch outputs.
- Preserve dense transform formulas and their existing host-math reproducibility boundary.
- Keep `RNGState<Engine>` as the concrete adapter and `DefaultRNGState = RNGState<DefaultRNG>`.
- Name the structural state concept `GeneratorState`; do not retain `CounterBasedRNGState` as a compatibility alias.
- In RandBLAS library headers, spell state templates as `GeneratorState state_t = DefaultRNGState` and do not redundantly qualify names with `RandBLAS::`.
- Tests, examples, and downstream code may use `RandBLAS::GeneratorState` where required by their namespace.
- Implement `RNGState`, `rng::Philox`, and `rng::RepackedOutput` as structs with no private members. The concrete adapter exposes `counter`, `key`, and `engine`; the repacker exposes `engine`; each uses memberwise defaulted equality when its member types support it.
- Keep `rng::CounterBasedEngine`, `rng::SeedMappableEngine`, and `GeneratorState` structural. Do not require inheritance or virtual dispatch.
- Keep `RNGStream` under `RandBLAS::testing::detail`; it is test-data infrastructure, not a production scalar RNG API.
- Keep only `rng::u01`, `rng::boxmuller`, `rng::uneg11`, and `rng::boxmul` as the supported distribution names. Delete `u01_block`, `uneg11_block`, and `boxmuller_block`.
- Retain D. E. Shaw Research's BSD-3-Clause notice verbatim in adapted files and add RandBLAS's 2026 copyright statement.
- Reflow code changed by these tasks for readability, but do not apply an arbitrary line-length limit or churn unrelated legacy code.
- Do not implement machine-specific optimization in this remediation. Document the opportunities in the PR description and require benchmarks before future optimization.
- Do not modify RandLAPACK.
- Do not push. The user controls publication of branch commits unless they explicitly delegate it.
- Preserve the pre-existing untracked `.claude/` directory and all unrelated user changes.
- This remediation plan supersedes conflicting naming, access-control, and distribution-helper statements in `docs/superpowers/specs/2026-07-31-native-cbrng-design.md` and `docs/superpowers/plans/2026-08-01-native-cbrng.md`. All three temporary planning artifacts are removed in Task 7 before merge.

---

## Execution protocol

Before implementation, commit this plan so it can serve as the cross-session record:

```bash
cd /Users/riley/randnla/dev/repo-randblas
git add docs/superpowers/plans/2026-08-07-native-cbrng-review-remediation.md
git commit -m "docs: plan native CBRNG review remediation"
```

For each task:

1. Run `git status --short --branch` and verify that only `.claude/` plus the task's expected files are dirty.
2. Add or update the focused test first. Observe the stated failure, or record the existing passing characterization when the task is a behavior-preserving refactor.
3. Make the smallest implementation change that satisfies the task.
4. Run the focused test, then the task-level regression command.
5. Run `git diff --check` and the task's source scan.
6. Commit only the task's files with the listed message.

## Review-to-task map

| Review point | Resolution | Owning task |
|---|---|---|
| State concept name and template spelling | Rename the concept to `GeneratorState`; retain concrete `RNGState`; use lower-case `state_t` without in-library `RandBLAS::` qualification | Task 1 |
| `class` and `private` in new RNG types | Convert all three to transparent structs; move Philox helpers to `rng::detail` | Task 1 |
| Distribution header readability | Remove local concepts and block helpers; retain scalar transforms and direct policy loops | Task 2 |
| `CBRNGStream` placement and role | Move to `RandBLAS/testing/rng.hh`, rename `RNGStream`, test buffering directly, document test-only status | Task 3 |
| TLS seed comments | Use `std::uint64_t` and restore the original constructor form | Task 4 |
| Random123-derived copyright | Add RandBLAS's 2026 statement without altering the D. E. Shaw notice | Task 4 |
| Unsupported standard-library claim | Delete the claim rather than add an evidence burden | Task 4 |
| FAQ correction | Link the templating statement to `GeneratorState`, and expose the concept in the API page | Task 4 |
| Line wrapping | Reflow only code touched by this remediation and perform a focused branch-added-code audit | Tasks 1-4, final audit in Task 5 |
| Deferred optimization notes | Add a concrete section to PR 182's description after local verification | Task 6 |
| Temporary design/plan artifacts | Remove the original design, original execution plan, and this remediation plan before merge | Task 7 |

---

### Task 1: Introduce `GeneratorState` and transparent RNG structs

**Files:**

- Create: `RandBLAS/rng/concepts.hh`
- Modify: `RandBLAS/random_gen.hh`
- Modify: `RandBLAS/rng/philox.hh`
- Modify: `RandBLAS/rng/repacked_output.hh`
- Modify: `RandBLAS/base.hh`
- Modify: `RandBLAS/dense_skops.hh`
- Modify: `RandBLAS/sparse_skops.hh`
- Modify: `RandBLAS/util.hh`
- Modify: `RandBLAS/testing/lapack_like.hh`
- Modify: `RandBLAS/testing/linops.hh`
- Modify: `RandBLAS/testing/sparse_data.hh`
- Modify: `examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc`
- Modify: `test/basic_rng/benchmark_speed.cc`
- Modify: `test/basic_rng/test_discrete.cc`
- Modify: `test/basic_rng/test_repacked_output.cc`
- Modify: `test/basic_rng/test_rng_state.cc`
- Modify: `test/basic_rng/test_sampler_regression.cc`
- Modify: `test/datastructures/test_denseskop.cc`

**Interfaces:**

- Produces: `RandBLAS::rng::CounterBasedEngine<engine_t>`.
- Produces: `RandBLAS::rng::SeedMappableEngine<engine_t>`.
- Produces: `RandBLAS::GeneratorState<state_t>`.
- Produces: public `RNGState::counter`, `RNGState::key`, and `RNGState::engine` data.
- Produces: public `RepackedOutput::engine` data.
- Preserves: `RNGState::generate(res_t&) const`, `RNGState::advance(uint64_t)`, all constructors, equality for the default and test states, and `DefaultRNGState`.

- [x] **Step 1: Add failing concept and public-data checks**

In `test/basic_rng/test_rng_state.cc`, replace the old concept assertion and accessor-only test with checks equivalent to:

```cpp
using OpaqueState = RandBLAS::RNGState<OpaqueEngine>;

template <class state_t>
concept HasPublicStateData = requires(state_t state) {
    state.counter;
    state.key;
    state.engine;
};

static_assert(RandBLAS::GeneratorState<OpaqueState>);
static_assert(HasPublicStateData<OpaqueState>);
static_assert(std::equality_comparable<OpaqueState>);

TEST(RNGState, ExposesItsValueStateAsPublicData) {
    OpaqueState state(UINT64_C(0x0123456789abcdef));
    EXPECT_EQ(state.counter, OpaqueCounter{});
    EXPECT_EQ(state.key,
              (OpaqueEngine::key_t{UINT32_C(0x89abcdef)}));
}
```

Add a defaulted equality operator to the test-only `OpaqueEngine` so the state
test exercises memberwise equality across counter, key, and engine.

In `test/basic_rng/test_repacked_output.cc`, add:

```cpp
template <class engine_t>
concept HasPublicWrappedEngine = requires(engine_t engine) {
    engine.engine;
};

using PublicRepacked =
    RandBLAS::rng::RepackedOutput<FixedEngine, std::uint16_t>;
static_assert(HasPublicWrappedEngine<PublicRepacked>);
static_assert(std::equality_comparable<
              RandBLAS::rng::Philox<4, 32, 10>>);
static_assert(std::equality_comparable<
              RandBLAS::rng::RepackedOutput<
                  RandBLAS::rng::Philox<4, 32, 10>, std::uint16_t>>);
```

Update existing state assertions in `test_rng_state.cc`, `test_discrete.cc`, `test_sampler_regression.cc`, and `test_denseskop.cc` from `state.counter()`/`state.key()` to `state.counter`/`state.key`.

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j stat_tests densedata_tests
```

Expected: compilation fails because `GeneratorState`, public `counter`/`key`/`engine`, and public repacker `engine` do not yet exist.

- [x] **Step 2: Extract the structural concepts**

Create `RandBLAS/rng/concepts.hh` with RandBLAS's standard license header and these definitions moved out of `random_gen.hh`:

```cpp
namespace RandBLAS::rng::detail {

template <class block_t>
concept FixedUnsignedBlock = requires {
    typename block_t::value_type;
    requires std::unsigned_integral<typename block_t::value_type>;
    requires(std::tuple_size_v<block_t> > 0);
};

} // namespace RandBLAS::rng::detail

namespace RandBLAS::rng {

template <class engine_t>
concept CounterBasedEngine =
    std::semiregular<engine_t> && requires {
        typename engine_t::ctr_t;
        typename engine_t::key_t;
        typename engine_t::res_t;
        requires std::regular<typename engine_t::ctr_t>;
        requires std::regular<typename engine_t::key_t>;
        requires detail::FixedUnsignedBlock<typename engine_t::res_t>;
    } && requires(engine_t const& engine,
                  typename engine_t::ctr_t& counter,
                  typename engine_t::ctr_t const& const_counter,
                  typename engine_t::key_t const& key,
                  typename engine_t::res_t& output,
                  std::uint64_t blocks) {
        { counter.advance(blocks) } -> std::same_as<void>;
        { engine.generate(const_counter, key, output) } ->
            std::same_as<void>;
    };

template <class engine_t>
concept SeedMappableEngine =
    CounterBasedEngine<engine_t> && requires(std::uint64_t seed) {
        { engine_t::make_key(seed) } ->
            std::same_as<typename engine_t::key_t>;
    };

} // namespace RandBLAS::rng

namespace RandBLAS {

template <class state_t>
concept GeneratorState =
    std::copyable<state_t> && requires {
        typename state_t::res_t;
        requires rng::detail::FixedUnsignedBlock<typename state_t::res_t>;
    } && requires(state_t& state, state_t const& const_state,
                  typename state_t::res_t& output, std::uint64_t blocks) {
        { const_state.generate(output) } -> std::same_as<void>;
        { state.advance(blocks) } -> std::same_as<void>;
    };

} // namespace RandBLAS
```

Copy the complete current `CounterBasedEngine` requirements, including counter advancement, fixed unsigned output, value semantics, and output-only generation. Include only `<concepts>`, `<cstdint>`, and `<tuple>`. Include `rng/concepts.hh` from `random_gen.hh` and delete the moved definitions from the umbrella header.

- [x] **Step 3: Make `RNGState` a transparent struct**

Change the concrete adapter to this public representation:

```cpp
template <rng::CounterBasedEngine Engine = DefaultRNG>
struct RNGState {
    using engine_t = Engine;
    using ctr_t = typename Engine::ctr_t;
    using key_t = typename Engine::key_t;
    using res_t = typename Engine::res_t;

    ctr_t counter{};
    key_t key{};
    [[no_unique_address]] Engine engine{};

    constexpr RNGState() = default;

    constexpr RNGState(std::uint64_t seed) noexcept(
        noexcept(Engine::make_key(seed)))
        requires rng::SeedMappableEngine<Engine>
        : key(Engine::make_key(seed)) {}

    explicit constexpr RNGState(key_t const& initial_key)
        : key(initial_key) {}

    constexpr RNGState(ctr_t const& initial_counter,
                       key_t const& initial_key)
        : counter(initial_counter), key(initial_key) {}

    constexpr void generate(res_t& output) const noexcept(
        noexcept(engine.generate(counter, key, output))) {
        engine.generate(counter, key, output);
    }

    constexpr void advance(std::uint64_t blocks) noexcept(
        noexcept(counter.advance(blocks))) {
        counter.advance(blocks);
    }

    friend constexpr bool operator==(RNGState const& left,
                                     RNGState const& right) = default;
};
```

Initialize and use the public members in every constructor and method. Remove
`counter()`, `key()`, the underscored member names, and the private section.
Defaulted equality compares all three value members and is conditionally
available when the engine supports equality; `GeneratorState` does not require
equality from arbitrary custom states.

Update `RandBLAS/base.hh`'s stream insertion operator to inspect `s.counter` and `s.key` directly.

- [x] **Step 4: Make Philox and repacking transparent structs**

Change `rng::Philox<N, W, R>` from `class` to `struct`. Move its implementation helpers into `RandBLAS::rng::detail` with these names:

```cpp
template <std::size_t N, std::size_t W>
struct PhiloxConstants;

template <std::size_t N, std::size_t W, class word_t, class key_t>
constexpr void apply_philox_round(std::array<word_t, N>& block,
                                  key_t const& key) noexcept;

template <std::size_t N, std::size_t W, class key_t>
constexpr void bump_philox_key(key_t& key) noexcept;
```

`PhiloxConstants` owns the multiplier and Weyl constants now returned by private member functions. `Philox::generate` calls the two detail functions and otherwise retains its current loop and output assignment. Keep `mulhilo` in `rng::detail`. The public `Philox` struct has only its compile-time validation, aliases, `make_key`, and `generate`.

Add memberwise equality to the stateless public type:

```cpp
friend constexpr bool operator==(Philox const&, Philox const&) = default;
```

Change `rng::RepackedOutput` from `class` to `struct`. Keep its aliases and compile-time metadata public and replace `engine_` with:

```cpp
[[no_unique_address]] Engine engine{};
```

Use `engine` in construction, `noexcept` expressions, and generation. Remove its private section.

Add memberwise equality to the repacker:

```cpp
friend constexpr bool operator==(RepackedOutput const&,
                                 RepackedOutput const&) = default;
```

This comparison is available when the wrapped engine is equality-comparable;
the engine concept itself remains only semiregular.

Include `concepts.hh` from `repacked_output.hh`, constrain the adapter with
`CounterBasedEngine<Engine>`, and delete the duplicate
`detail::EngineHasFixedUnsignedResult` concept. Keep `ValidRepacking` as the
separate width-ratio constraint.

- [x] **Step 5: Rename the state concept and template parameter throughout code**

Apply these exact vocabulary rules:

```cpp
template <typename T, GeneratorState state_t = DefaultRNGState>
struct DenseSkOp;

template <typename T, SignedInteger sint_t,
          GeneratorState state_t = DefaultRNGState>
state_t sample_indices_iid(std::int64_t n, T const* cdf, std::int64_t k,
                           sint_t* samples, state_t const& state);
```

Within RandBLAS headers, replace template parameter `State` with `state_t` and update parameter/member type uses in the same declaration or definition. Remove `RandBLAS::` qualification from `GeneratorState` and `DefaultRNGState` inside `namespace RandBLAS` and nested `RandBLAS::*` namespaces.

In tests and examples outside namespace RandBLAS, replace the concept name with `RandBLAS::GeneratorState`; retaining a local capitalized template parameter there is allowed. Do not add a `CounterBasedRNGState` alias.

- [x] **Step 6: Run focused and full tests**

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j stat_tests densedata_tests sparsedata_tests meta_tests misc_tests test_rng_speed
ctest --output-on-failure -R 'RNGState|Philox|RepackedOutput|SamplerRegression'
ctest --output-on-failure
```

Expected: all targets compile and every test passes. State equality, KATs, repacking, sampler regression, thread-count independence, and state-advance tests remain unchanged in behavior.

- [x] **Step 7: Audit vocabulary, access control, and formatting**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'CounterBasedRNGState' RandBLAS test examples --glob '*.{hh,cc}'
rg -n 'RandBLAS::GeneratorState|RandBLAS::DefaultRNGState' RandBLAS --glob '*.hh'
rg -n 'GeneratorState State|class (RNGState|Philox|RepackedOutput)|private:' RandBLAS/random_gen.hh RandBLAS/rng/philox.hh RandBLAS/rng/repacked_output.hh
rg -n '\.counter\(\)|\.key\(\)' RandBLAS test examples
git diff --check
```

Expected: all five scans have no matches. Manually inspect the task diff and join wrapped expressions that now fit comfortably on one readable line; do not reformat unrelated code.

- [x] **Step 8: Commit the public API remediation**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git add RandBLAS/rng/concepts.hh RandBLAS/random_gen.hh RandBLAS/rng/philox.hh RandBLAS/rng/repacked_output.hh RandBLAS/base.hh RandBLAS/dense_skops.hh RandBLAS/sparse_skops.hh RandBLAS/util.hh RandBLAS/testing/lapack_like.hh RandBLAS/testing/linops.hh RandBLAS/testing/sparse_data.hh examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc test/basic_rng/benchmark_speed.cc test/basic_rng/test_discrete.cc test/basic_rng/test_repacked_output.cc test/basic_rng/test_rng_state.cc test/basic_rng/test_sampler_regression.cc test/datastructures/test_denseskop.cc
git commit -m "refactor: simplify native RNG state interfaces"
```

---

### Task 2: Simplify floating-point distribution transforms

**Files:**

- Modify: `RandBLAS/rng/distributions.hh`
- Modify: `test/basic_rng/test_distributions.cc`

**Interfaces:**

- Consumes: `GeneratorState<state_t>` from `RandBLAS/rng/concepts.hh`.
- Preserves: `rng::u01<Real>(word)`, `rng::boxmuller(angle_word, radius_word)`, `rng::uneg11::convert<Real>(word)`, `rng::uneg11::generate(state)`, and `rng::boxmul::generate(state)`.
- Removes: `rng::u01_block`, `rng::uneg11_block`, and `rng::boxmuller_block`.

- [x] **Step 1: Record the passing scalar and policy characterization**

Run before editing:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j stat_tests
ctest --output-on-failure -R 'Distribution|Continuous|SamplerRegression'
```

Expected: the current scalar reference, endpoint, policy, continuous-statistical, and sampler-regression tests pass. This is the behavior oracle for the readability refactor.

- [x] **Step 2: Rewrite tests around the supported API**

In `test/basic_rng/test_distributions.cc`:

- add a `std::uint64_t blocks{}` member and
  `void advance(std::uint64_t amount) { blocks += amount; }` to `FixedState` so
  it satisfies `GeneratorState`;
- assert `RandBLAS::GeneratorState<FixedState<std::uint32_t, 4>>`;
- delete `BlockHelpersPreserveLengthAndLaneMapping`;
- preserve every scalar reference and endpoint test;
- change the policy test to compare each uniform result directly with `uneg11::convert` and each adjacent normal pair directly with `boxmuller`.

Use this comparison shape:

```cpp
auto uniform = RandBLAS::rng::uneg11::generate(state);
auto normal = RandBLAS::rng::boxmul::generate(state);

for (std::size_t i = 0; i < bits.size(); ++i) {
    EXPECT_EQ(uniform[i],
              RandBLAS::rng::uneg11::convert<float>(bits[i]));
}
for (std::size_t i = 0; i < bits.size(); i += 2) {
    auto pair = RandBLAS::rng::boxmuller(bits[i], bits[i + 1]);
    EXPECT_EQ(normal[i], pair[0]);
    EXPECT_EQ(normal[i + 1], pair[1]);
}
```

Retain the compile-time rejection of an odd Box--Muller result length.

- [x] **Step 3: Remove the local concept layer and block helpers**

Include `concepts.hh` from `distributions.hh`. Delete these local concepts:

- `SupportedDistributionWord`;
- `SupportedDistributionReal`; and
- `StateCanGenerateFixedUnsignedBlock`.

Delete all overloads of `u01_block`, `uneg11_block`, and `boxmuller_block`. Keep one small `detail::default_real_t<word_t>` alias mapping 32-bit words to `float` and 64-bit words to `double`.

Write scalar templates with ordinary type parameters and adjacent assertions:

```cpp
template <typename real_t, typename word_t>
[[nodiscard]] constexpr real_t u01(word_t input) noexcept {
    static_assert(std::is_unsigned_v<word_t>);
    static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
    static_assert(std::is_same_v<real_t, float> ||
                  std::is_same_v<real_t, double>);
    constexpr real_t factor =
        real_t{1} /
        (static_cast<real_t>(std::numeric_limits<word_t>::max()) + real_t{1});
    constexpr real_t half_factor = real_t{0.5} * factor;
    return static_cast<real_t>(input) * factor + half_factor;
}
```

Implement the retained scalar transforms directly:

```cpp
template <typename word_t>
using default_real_t =
    std::conditional_t<sizeof(word_t) == 4, float, double>;

struct uneg11 {
    template <typename real_t, typename word_t>
    [[nodiscard]] static constexpr real_t convert(word_t input) noexcept {
        static_assert(std::is_unsigned_v<word_t>);
        static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
        static_assert(std::is_same_v<real_t, float> ||
                      std::is_same_v<real_t, double>);
        using signed_word_t = std::make_signed_t<word_t>;
        constexpr real_t factor =
            real_t{1} /
            (static_cast<real_t>(
                 std::numeric_limits<signed_word_t>::max()) + real_t{1});
        constexpr real_t half_factor = real_t{0.5} * factor;
        return static_cast<real_t>(static_cast<signed_word_t>(input)) * factor +
               half_factor;
    }

    template <GeneratorState state_t>
    [[nodiscard]] static auto generate(state_t const& state) {
        using bits_t = typename state_t::res_t;
        using word_t = typename bits_t::value_type;
        using real_t = detail::default_real_t<word_t>;
        constexpr std::size_t count = std::tuple_size_v<bits_t>;
        bits_t bits{};
        std::array<real_t, count> output{};
        state.generate(bits);
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = convert<real_t>(bits[i]);
        }
        return output;
    }
};

template <typename word_t>
[[nodiscard]] inline auto boxmuller(word_t angle_word, word_t radius_word) {
    static_assert(std::is_unsigned_v<word_t>);
    static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
    using real_t = detail::default_real_t<word_t>;
    constexpr real_t pi = real_t{3.1415926535897932};
    auto angle = pi * uneg11::convert<real_t>(angle_word);
    auto radius =
        std::sqrt(real_t{-2} * std::log(u01<real_t>(radius_word)));
    return std::array<real_t, 2>{std::sin(angle) * radius,
                                 std::cos(angle) * radius};
}
```

Do not change constants, casts, endpoints, word assignment, or math functions.

- [x] **Step 4: Put straightforward loops in the two policies**

Implement the policies with the public state concept and direct loops:

```cpp
struct uneg11 {
    template <typename real_t, typename word_t>
    [[nodiscard]] static constexpr real_t convert(word_t input) noexcept {
        static_assert(std::is_unsigned_v<word_t>);
        static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
        static_assert(std::is_same_v<real_t, float> ||
                      std::is_same_v<real_t, double>);
        using signed_word_t = std::make_signed_t<word_t>;
        constexpr real_t factor =
            real_t{1} /
            (static_cast<real_t>(
                 std::numeric_limits<signed_word_t>::max()) + real_t{1});
        constexpr real_t half_factor = real_t{0.5} * factor;
        return static_cast<real_t>(static_cast<signed_word_t>(input)) * factor +
               half_factor;
    }

    template <GeneratorState state_t>
    [[nodiscard]] static auto generate(state_t const& state) {
        using bits_t = typename state_t::res_t;
        using word_t = typename bits_t::value_type;
        using real_t = detail::default_real_t<word_t>;
        constexpr std::size_t count = std::tuple_size_v<bits_t>;
        bits_t bits{};
        std::array<real_t, count> output{};
        state.generate(bits);
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = convert<real_t>(bits[i]);
        }
        return output;
    }
};

struct boxmul {
    template <GeneratorState state_t>
        requires(std::tuple_size_v<typename state_t::res_t> % 2 == 0)
    [[nodiscard]] static auto generate(state_t const& state) {
        using bits_t = typename state_t::res_t;
        using word_t = typename bits_t::value_type;
        using real_t = detail::default_real_t<word_t>;
        constexpr std::size_t count = std::tuple_size_v<bits_t>;
        bits_t bits{};
        std::array<real_t, count> output{};
        state.generate(bits);
        for (std::size_t i = 0; i < count; i += 2) {
            auto pair = boxmuller(bits[i], bits[i + 1]);
            output[i] = pair[0];
            output[i + 1] = pair[1];
        }
        return output;
    }
};
```

Each `generate` obtains one `res_t` block exactly once and never advances the input state. `uneg11::generate` loops over individual lanes and calls `convert`; `boxmul::generate` loops by two and calls `boxmuller`. Return `std::array<default_real_t<word_t>, N>` where `N` is the state result extent.

- [x] **Step 5: Verify behavior and reduced surface area**

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j stat_tests densedata_tests sparsedata_tests
ctest --output-on-failure -R 'Distribution|Continuous|SamplerRegression'
ctest --output-on-failure
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'SupportedDistributionWord|SupportedDistributionReal|StateCanGenerateFixedUnsignedBlock|u01_block|uneg11_block|boxmuller_block' RandBLAS test examples rtd
git diff --check
```

Expected: all tests pass and the source scan has no matches. The scalar formulas should be visually dominant in the final header.

- [x] **Step 6: Commit the distribution refactor**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git add RandBLAS/rng/distributions.hh test/basic_rng/test_distributions.cc
git commit -m "refactor: simplify native RNG distributions"
```

---

### Task 3: Extract and test the test-only scalar stream

**Files:**

- Create: `RandBLAS/testing/rng.hh`
- Create: `test/meta/test_rng_stream.cc`
- Modify: `RandBLAS/testing/sparse_data.hh`
- Modify: `test/CMakeLists.txt`
- Modify: `test/DevNotes.md`

**Interfaces:**

- Produces: `RandBLAS::testing::detail::RNGStream<state_t>`.
- Preserves: `next_word`, `uniform_01`, `gaussian<T>`, `geometric`, and `get_state` behavior.
- Preserves: fetching a new result block advances the held state immediately by one block, even when buffered lanes remain unread.

- [x] **Step 1: Add a failing focused stream test**

Create `test/meta/test_rng_stream.cc` and add it to `META_SOURCES`. Define a deterministic state:

```cpp
struct SequenceState {
    using res_t = std::array<std::uint32_t, 2>;

    res_t first{};
    std::uint64_t block{};

    void generate(res_t& output) const {
        output = {
            static_cast<std::uint32_t>(first[0] + 2 * block),
            static_cast<std::uint32_t>(first[1] + 2 * block)
        };
    }

    void advance(std::uint64_t blocks) { block += blocks; }
};

static_assert(RandBLAS::GeneratorState<SequenceState>);
```

Add three tests:

1. `NextWordBuffersOneBlockAndAdvancesOnRefill`: consume three words, verify the first two come from block zero, the third comes from block one, and `get_state().block` changes from one to two only at refills.
2. `GaussianCachesTheSecondValue`: initialize the first two words to `0x243f6a88` and `0x85a308d3`, compare two calls with one `rng::boxmuller` call, and verify the second call neither generates nor advances another block.
3. `UniformAndGeometricUseScalarConversions`: use separate freshly constructed streams, compare `uniform_01` with `rng::u01<double>`, and compare `geometric(log(0.75))` with the same explicit inverse-CDF expression used by the helper.

Initially include `RandBLAS/testing/rng.hh` and refer to `RandBLAS::testing::detail::RNGStream<SequenceState>`.

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j meta_tests
```

Expected: compilation fails because `RandBLAS/testing/rng.hh` and `RNGStream` do not exist.

- [x] **Step 2: Move and rename the helper**

Create `RandBLAS/testing/rng.hh` with RandBLAS's standard license header. Include
`RandBLAS/random_gen.hh`, `<cmath>`, `<cstddef>`, `<cstdint>`, and `<tuple>`.
Move `CBRNGStream` out of `sparse_data.hh`, rename it `RNGStream`, and use this
complete definition:

```cpp
namespace RandBLAS::testing::detail {

template <GeneratorState state_t = DefaultRNGState>
struct RNGStream {
    using res_t = typename state_t::res_t;
    using word_t = typename res_t::value_type;
    static constexpr std::size_t block_size = std::tuple_size_v<res_t>;

    state_t state;
    res_t buffer{};
    std::size_t pos = block_size;
    double spare = 0.0;
    bool has_spare = false;

    explicit RNGStream(state_t const& initial_state)
        : state(initial_state) {}

    word_t next_word() {
        if (pos >= block_size) {
            state.generate(buffer);
            state.advance(1);
            pos = 0;
        }
        return buffer[pos++];
    }

    double uniform_01() {
        return rng::u01<double>(next_word());
    }

    template <typename value_t>
    value_t gaussian() {
        if (has_spare) {
            has_spare = false;
            return static_cast<value_t>(spare);
        }
        word_t angle_word = next_word();
        word_t radius_word = next_word();
        auto [first, second] = rng::boxmuller(angle_word, radius_word);
        spare = second;
        has_spare = true;
        return static_cast<value_t>(first);
    }

    std::int64_t geometric(double log_1_minus_p) {
        double u = uniform_01();
        return static_cast<std::int64_t>(
            std::floor(std::log(1.0 - u) / log_1_minus_p));
    }

    state_t get_state() const { return state; }
};

} // namespace RandBLAS::testing::detail
```

Keep the existing algorithms and consumption order. Reflow the comments to state the contracts directly. In particular, document that `get_state()` reports the state after every block already loaded into the buffer, not after an abstract fractional block position.

- [x] **Step 3: Rewire sparse test-data generation**

Include `RandBLAS/testing/rng.hh` from `RandBLAS/testing/sparse_data.hh`. Remove the old helper definition and replace all three `detail::CBRNGStream<state_t>` uses with `detail::RNGStream<state_t>`. Remove `<tuple>` from `sparse_data.hh` after confirming it has no remaining use; retain `<cmath>` because sparse generation itself computes logarithms.

Add this permanent note under a new `### RNG stream` subsection in `test/DevNotes.md`:

```markdown
`RandBLAS/testing/rng.hh` contains the test-only `detail::RNGStream` adapter.
It turns fixed result blocks into a sequential word stream for random sparse
test-matrix generation and supplies the uniform, Gaussian, and geometric draws
needed there. Loading a block advances its held state immediately; unread lanes
remain in its local buffer. Production RandBLAS sampling remains
coordinate-addressed and does not use this sequential adapter.
```

- [x] **Step 4: Verify the stream and sparse generators**

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j meta_tests sparsedata_tests
ctest --output-on-failure -R 'RNGStream|RandomSparseMatrix|Sparse'
ctest --output-on-failure
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'CBRNGStream' RandBLAS test examples rtd
rg -n 'RNGStream' RandBLAS test
git diff --check
```

Expected: all tests pass; the first scan has no matches; the second scan is limited to `RandBLAS/testing/rng.hh`, its direct test, the three sparse-data uses, and `test/DevNotes.md`.

- [x] **Step 5: Commit the test-infrastructure extraction**

```bash
cd /Users/riley/randnla/dev/repo-randblas
git add RandBLAS/testing/rng.hh RandBLAS/testing/sparse_data.hh test/meta/test_rng_stream.cc test/CMakeLists.txt test/DevNotes.md
git commit -m "test: isolate the sequential RNG stream"
```

---

### Task 4: Resolve licensing, examples, documentation, and readability comments

**Files:**

- Modify: `RandBLAS/random_gen.hh`
- Modify: `RandBLAS/rng/concepts.hh`
- Modify: `RandBLAS/rng/philox.hh`
- Modify: `RandBLAS/rng/distributions.hh`
- Modify: `RandBLAS/rng/repacked_output.hh`
- Modify: `RandBLAS/testing/rng.hh`
- Modify: `test/basic_rng/philox_kat_vectors.txt`
- Modify: `test/basic_rng/test_rng_state.cc`
- Modify: `examples/total-least-squares/tls_dense_skop.cc`
- Modify: `examples/total-least-squares/tls_sparse_skop.cc`
- Modify: `RandBLAS/rng/DevNotes.md`
- Modify: `test/DevNotes.md`
- Modify: `rtd/source/FAQ.rst`
- Modify: `rtd/source/api_reference/skops_and_dists.rst`
- Modify: `rtd/source/tutorial/distributions.rst`
- Modify: `rtd/source/tutorial/sampling_skops.rst`
- Modify: `rtd/source/tutorial/sketch_updates.rst`

**Interfaces:**

- Preserves: RNG output and consumption behavior.
- Restores: implicit construction of the default state from a scalar seed, as
  required by the natural sketch-operator examples and supported before this PR.
- Documents: `GeneratorState`, transparent concrete state data, supported distribution names, test-only `RNGStream`, provenance, and deferred optimization scope.

- [x] **Step 1: Record the failing review scan**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'CounterBasedRNGState|counter\(\)|key\(\)|Standard-library distributions are not substituted' RandBLAS/rng/DevNotes.md test/DevNotes.md rtd
rg -n 'uint32_t seed = 1997|DefaultRNGState\{seed\}' examples/total-least-squares
head -n 4 RandBLAS/rng/philox.hh RandBLAS/rng/distributions.hh test/basic_rng/philox_kat_vectors.txt
```

Expected: the first two scans show the reviewed stale text and constructor form; the three adapted files show only the D. E. Shaw copyright at their start.

- [x] **Step 2: Add dual copyright attribution**

Prepend this line, using the file's comment syntax, to the two adapted headers and the KAT fixture:

```text
Copyright, 2026. See LICENSE for copyright holder information.
```

Use `//` in `.hh` files and `#` in the vector file. Leave the complete D. E. Shaw Research notice byte-for-byte unchanged immediately below the new statement. Do not add RandBLAS attribution to `word_array.hh` or `repacked_output.hh`; they already carry the standard RandBLAS header and are not dual-license fixes.

- [x] **Step 3: Restore the natural TLS constructor examples**

In both total-least-squares examples, include `<cstdint>` directly if the file does not already own that include, then use:

```cpp
std::uint64_t seed = 1997;
RandBLAS::DenseSkOp<double> S(Dist, seed);
```

and:

```cpp
std::uint64_t seed = 1997;
RandBLAS::SparseSkOp<double> S(Dist, seed);
```

Remove the explicit `DefaultRNGState{seed}` construction. Keep each constructor invocation on one line.

- [x] **Step 4: Correct permanent RNG and test notes**

Update `RandBLAS/rng/DevNotes.md` as follows:

- name the structural concept `GeneratorState` everywhere;
- describe `RNGState<Engine>` as the provided transparent adapter with public `counter`, `key`, and `engine` values;
- keep clear that generic code depends only on `generate`/`advance` and does not require those public members;
- update the standard-library comparison row from `CounterBasedRNGState` to `GeneratorState`;
- rewrite the distribution comparison row to say only that RandBLAS transforms
  explicit words/result blocks without owning or mutating generator state;
- remove the sentence claiming standard-library mappings and consumption patterns are not portable and may cache or vary consumption;
- list only `u01`, `boxmuller`, `uneg11`, and `boxmul` as supported transform names;
- describe `RNGStream` only as test infrastructure and point to `test/DevNotes.md` for its consumption details;
- preserve the Philox paper, Random123 revision, BSD provenance, reproducibility, and validation statements that remain factual.

Update `test/DevNotes.md` so the RNG-state entry says public data rather than const accessors and the distribution entry no longer refers to block helpers.

- [x] **Step 5: Correct public documentation and API links**

Change the FAQ sentence to:

```rst
 * Templates. We template for floating point precision just about everywhere.
   Sampling functions and sketching operators also template on random-number
   state types satisfying :cpp:any:`RandBLAS::GeneratorState`, and on arrays
   of 32-bit versus 64-bit signed integers.
```

In `rtd/source/api_reference/skops_and_dists.rst`, add a `GeneratorState` dropdown containing:

```rst
    .. doxygenconcept:: RandBLAS::GeneratorState
        :project: RandBLAS
```

Keep the existing `RNGState` struct dropdown separately. Replace stale `CounterBasedRNGState` tutorial comments with `GeneratorState`. In `sampling_skops.rst`, replace the const `counter()`/`key()` accessor description with public `counter`/`key` data and state that generic samplers require only the `GeneratorState` operations.

- [x] **Step 6: Perform the focused readability pass**

Review the branch-added RNG files and the files changed in Tasks 1-4:

```bash
cd /Users/riley/randnla/dev/repo-randblas
git diff origin/main -- RandBLAS/random_gen.hh RandBLAS/rng RandBLAS/testing/rng.hh RandBLAS/testing/sparse_data.hh test/basic_rng test/meta/test_rng_stream.cc examples/total-least-squares rtd/source/FAQ.rst rtd/source/api_reference/skops_and_dists.rst rtd/source/tutorial/distributions.rst rtd/source/tutorial/sampling_skops.rst rtd/source/tutorial/sketch_updates.rst
```

Join declarations, expressions, and short comments that were split solely to satisfy a narrow line budget. Retain line breaks that expose algorithm structure, separate template constraints, or keep tables and prose readable. Do not run a bulk formatter over the repository.

- [x] **Step 7: Build examples and documentation**

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j install
cd /Users/riley/randnla/dev/build-randblas-examples
make -j tls_dense_skop tls_sparse_skop
cd /Users/riley/randnla/dev/repo-randblas/rtd
sphinx-build source build
```

Expected: the library installs, both TLS targets compile with the scalar seed constructor, and Sphinx/Doxygen completes without a new missing-symbol warning for `GeneratorState`.

- [x] **Step 8: Verify the review fixes and commit**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'CounterBasedRNGState|counter\(\)|key\(\)|Standard-library distributions are not substituted|u01_block|uneg11_block|boxmuller_block' RandBLAS test examples rtd
rg -n 'uint32_t seed = 1997|DefaultRNGState\{seed\}' examples/total-least-squares
head -n 4 RandBLAS/rng/philox.hh RandBLAS/rng/distributions.hh test/basic_rng/philox_kat_vectors.txt
git diff --check
```

Expected: the first two scans have no matches. Each adapted file starts with RandBLAS's 2026 statement followed by the unchanged D. E. Shaw notice.

```bash
git add RandBLAS/rng/philox.hh RandBLAS/rng/distributions.hh test/basic_rng/philox_kat_vectors.txt examples/total-least-squares/tls_dense_skop.cc examples/total-least-squares/tls_sparse_skop.cc RandBLAS/rng/DevNotes.md test/DevNotes.md rtd/source/FAQ.rst rtd/source/api_reference/skops_and_dists.rst rtd/source/tutorial/distributions.rst rtd/source/tutorial/sampling_skops.rst rtd/source/tutorial/sketch_updates.rst
git commit -m "docs: address native RNG review feedback"
```

---

### Task 5: Run final local validation

**Files:**

- Verify: all files changed by PR 182.

**Interfaces:**

- Produces: clean build, test, installation, downstream, example, documentation, and performance evidence.

- [ ] **Step 1: Run the required workspace build and full test suite**

Run:

```bash
cd /Users/riley/randnla/dev/build-randblas
source sourceme.sh
make -j
ctest --output-on-failure
```

Expected: the complete configured build and every discovered test pass.

- [ ] **Step 2: Validate a clean Random123-disabled build and install**

Run these commands in one shell so the task-specific paths remain available:

```bash
cd /Users/riley/randnla/dev
source sourceme.sh
cbrng_review_build=$(mktemp -d /private/tmp/randblas-cbrng-review-build.XXXXXX)
cbrng_review_install=$(mktemp -d /private/tmp/randblas-cbrng-review-install.XXXXXX)
cmake -S repo-randblas -B "$cbrng_review_build" -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX="$cbrng_review_install" -Dblaspp_DIR=/Users/riley/randnla/dev/install-deps/lib/cmake/blaspp -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON
cmake --build "$cbrng_review_build" -j
ctest --test-dir "$cbrng_review_build" --output-on-failure
cmake --build "$cbrng_review_build" -j --target install
```

Expected: configure, compile, all tests, and installation succeed without
finding Random123. Confirm
`RandBLAS/rng/concepts.hh` and `RandBLAS/testing/rng.hh` are present below
`$cbrng_review_install/include/RandBLAS/`.

- [ ] **Step 3: Validate installed downstream and example builds**

Continue in the same shell:

```bash
cbrng_review_downstream=$(mktemp -d /private/tmp/randblas-cbrng-review-downstream.XXXXXX)
cmake -S repo-randblas/test/downstream -B "$cbrng_review_downstream" -DCMAKE_PREFIX_PATH="$cbrng_review_install" -Dblaspp_DIR=/Users/riley/randnla/dev/install-deps/lib/cmake/blaspp -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON
cmake --build "$cbrng_review_downstream" -j
"$cbrng_review_downstream/smoke"

cbrng_review_examples=$(mktemp -d /private/tmp/randblas-cbrng-review-examples.XXXXXX)
cmake -S repo-randblas/examples -B "$cbrng_review_examples" -DCMAKE_PREFIX_PATH="$cbrng_review_install" -Dblaspp_DIR=/Users/riley/randnla/dev/install-deps/lib/cmake/blaspp -Dlapackpp_DIR=/Users/riley/randnla/dev/install-deps/lib/cmake/lapackpp -DCMAKE_DISABLE_FIND_PACKAGE_Random123=ON -DFETCHCONTENT_SOURCE_DIR_FAST_MATRIX_MARKET=/Users/riley/randnla/dev/build-randblas-examples/_deps/fast_matrix_market-src
cmake --build "$cbrng_review_examples" -j
```

Expected: the installed-package smoke executable runs successfully and every example compiles without a Random123 package path.

- [ ] **Step 4: Re-run the native RNG performance smoke test**

Run seven trials with the same dimensions and thread count as the original validation:

```bash
for cbrng_trial in 1 2 3 4 5 6 7; do
    OMP_NUM_THREADS=1 "$cbrng_review_build/bin/test_rng_speed" 8192 1024
done
```

Expected: the measurements remain within the prior native run's ordinary variation; investigate any repeatable regression before committing the final cleanup.

- [ ] **Step 5: Run final source and whitespace audits**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'CounterBasedRNGState|CBRNGStream|u01_block|uneg11_block|boxmuller_block|class (RNGState|Philox|RepackedOutput)' RandBLAS test examples rtd
rg -n 'RandBLAS::GeneratorState|RandBLAS::DefaultRNGState' RandBLAS --glob '*.hh'
rg -n '\.counter\(\)|\.key\(\)' RandBLAS test examples rtd
rg -n 'Random123/|r123::|r123ext::|Random123_DIR|find_package\(Random123|Random123::Random123|R123_' . --glob '!rtd/source/updates/index.rst'
rg -n 'Copyright, 2026' RandBLAS/rng/philox.hh RandBLAS/rng/distributions.hh test/basic_rng/philox_kat_vectors.txt
git diff --check
git status --short --branch
```

Expected:

- the first three scans have no matches;
- the functional dependency scan has no matches, with historical/provenance prose inspected separately;
- all three adapted files contain RandBLAS's 2026 line and retain the D. E. Shaw notice;
- whitespace validation passes;
- the branch is clean except for the pre-existing untracked `.claude/` directory.

---

### Task 6: Update PR 182 and close the review loop

**External state:**

- Modify: PR 182 description.
- Reply: inline review threads `3701044665`, `3701044854`, `3701060511`, `3701212774`, and `3701293814`.
- Verify: PR 182 CI after the user publishes the local commits.

**Interfaces:**

- Consumes: the verified local commits from Tasks 1-5.
- Produces: a PR description that identifies deferred optimization work and review threads tied to verified resolutions.

- [ ] **Step 1: Hand the verified branch to the user for publication**

Report the commit list, local verification commands, and clean/dirty status. Do not run `git push`. Wait for the user to confirm that the commits are on `origin/native-cbrng` before changing review-thread state or monitoring CI.

- [ ] **Step 2: Replace the work-in-progress PR description**

Replace the current planning-era description with this complete body:

```markdown
This PR is a work in progress.

## Summary

- removes RandBLAS's source, package, CI, and installed-package dependency on
  Random123;
- provides native, header-only Philox engines with static known-answer tests;
- introduces the concrete `RNGState<Engine>` adapter and structural
  `GeneratorState` customization boundary;
- provides `RepackedOutput` for power-of-two output-word subdivision; and
- preserves coordinate-addressed, thread-count-independent sampling.

The default `Philox<4, 32, 10>` integer stream and default sparse-sketch output
remain bitwise compatible with the previous Random123-backed implementation.
Dense transforms retain the same formulas subject to host math-library rounding.

## Validation

The branch includes Philox known-answer tests, counter/repacking/transform unit
tests, statistical tests, sampler regression tests, installed downstream and
example builds, and clean builds with Random123 discovery disabled.

## Deferred optimization opportunities

This PR uses portable implementations and intentionally defers
architecture/compiler-specific tuning. Follow-up performance work could
evaluate:

- dedicated 64-bit multiply-high instructions or newer compiler builtins in
  Philox's `mulhilo` path;
- compiler-specific unrolling or vectorization pragmas for Philox rounds and
  repacked-output loops; and
- SIMD implementations of result-block floating-point transforms.

These changes should be benchmarked by compiler and architecture before they
replace the portable code.
```

Create `/private/tmp/randblas-pr-182-body.md` with `apply_patch`, using the exact body above, and run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
gh pr edit 182 --body-file /private/tmp/randblas-pr-182-body.md
gh pr view 182 --json body --jq .body
```

Delete the temporary body file with `apply_patch` after `gh pr edit` succeeds.
Expected: the rendered description contains the native CBRNG summary,
validation scope, and exact deferred-optimization section once; it no longer
refers to a future specification or plan.

- [ ] **Step 3: Reply to each inline review thread with the verified resolution**

Post these concise replies through the thread-reply endpoint:

```bash
gh api repos/BallisticLA/RandBLAS/pulls/182/comments/3701044665/replies -f body='Changed the seed to std::uint64_t and restored DenseSkOp<double> S(Dist, seed). The TLS example target compiles.'
gh api repos/BallisticLA/RandBLAS/pulls/182/comments/3701044854/replies -f body='Changed the seed to std::uint64_t and restored SparseSkOp<double> S(Dist, seed). The TLS example target compiles.'
gh api repos/BallisticLA/RandBLAS/pulls/182/comments/3701060511/replies -f body='Removed the unsupported standard-library distribution claim; the notes now state only RandBLAS contracts and verified behavior.'
gh api repos/BallisticLA/RandBLAS/pulls/182/comments/3701212774/replies -f body='Simplified the header around the scalar formulas: the three file-local concepts and public block helpers are gone, and the two policies use direct loops. Scalar references, policy tests, statistical tests, and sampler regressions pass.'
gh api repos/BallisticLA/RandBLAS/pulls/182/comments/3701293814/replies -f body='The FAQ now points to the GeneratorState concept, and the API page documents GeneratorState separately from the concrete RNGState adapter.'
```

Expected: each reply appears in its original inline thread rather than as a top-level PR comment.

- [ ] **Step 4: Monitor the published commit's CI**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
gh pr checks 182 --watch --interval 30
```

Expected: all required PR checks pass. If a check fails, capture its log, use the systematic-debugging workflow, make the smallest local fix with a focused regression test, rerun the relevant local verification, and return to Step 1 so the user can publish the additional commit.

---

### Task 7: Remove temporary planning artifacts before merge

**Files:**

- Delete: `docs/superpowers/specs/2026-07-31-native-cbrng-design.md`
- Delete: `docs/superpowers/plans/2026-08-01-native-cbrng.md`
- Delete: `docs/superpowers/plans/2026-08-07-native-cbrng-review-remediation.md`

**Interfaces:**

- Consumes: passing local validation and passing PR checks from Tasks 5-6.
- Produces: a merge-ready tree whose lasting rationale is confined to permanent developer and user documentation.

- [ ] **Step 1: Confirm permanent notes cover the retained rationale**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
rg -n 'GeneratorState|RNGState|RNGStream|Philox|RepackedOutput|provenance|license|known-answer|thread' RandBLAS/rng/DevNotes.md test/DevNotes.md rtd/source/tutorial/sampling_skops.rst
```

Expected: the permanent files cover the public contracts, transparent concrete state, test-only stream, stream/repacking semantics, provenance, and validation strategy without relying on a temporary plan.

- [ ] **Step 2: Delete and commit all temporary planning files**

Run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
git rm docs/superpowers/specs/2026-07-31-native-cbrng-design.md docs/superpowers/plans/2026-08-01-native-cbrng.md docs/superpowers/plans/2026-08-07-native-cbrng-review-remediation.md
git diff --check
git status --short --branch
git commit -m "docs: remove temporary native RNG plans"
```

Expected: the commit contains only the three deletions; `.claude/` remains untouched.

- [ ] **Step 3: Hand the final documentation-only commit to the user**

Report the new commit hash and do not push. After the user confirms it is published, run:

```bash
cd /Users/riley/randnla/dev/repo-randblas
gh pr checks 182 --watch --interval 30
```

Expected: all required checks pass on the final PR head, including the planning-artifact deletion commit.
