# Plan: a `sketch_general` benchmark, analogous to `spmm_performance.cc`

## Goal

`spmm_performance.cc` benchmarks the low-level `left_spmm` kernels on **generic
uniform-random** sparse matrices. It therefore never exercises the *structured*
sketching operators (fixed nnz per column/row, ±1 values, mostly-empty columns),
never hits the regular-CSC fast path, and never goes through `sketch_general`.
This benchmark closes that gap: it measures the **full sketching call**
(`sketch_general` with a real `SparseSkOp`) across the operator shapes identified
in `sparsesketching_opt.md`, so we get a baseline for the proposed kernels
(CountSketch, bounded-CSR for wide LASO, regular-path layout coverage, ±1
implicit-value) before any of them is written.

New executable: `examples/simple-kernel-benchmarks/sketch_general_performance.cc`.

## What `sketch_general` does, and the cost lifecycle we must measure

Left-sketch form (the headline case):

```
sketch_general(layout, opS, opA, d, n, m, alpha, S, ro_s, co_s, A, lda, beta, B, ldb)
   // B(d x n) = alpha * op(S)(d x m) * op(A)(m x n) + beta * B
```

with `S` a `SparseSkOp` of distribution `SparseDist(d, m, vec_nnz, major_axis)`.
For a sparse `S` this routes `lskges → coo_view_of_skop → left_spmm`. The cost has
**three distinct phases** that the benchmark must be able to separate, because the
opt-doc proposals target different ones:

1. **Sample / fill** — `fill_sparse(S)` populates `(rows, cols, vals)`. Happens
   once in real build-once/apply-many use; if `S.nnz < 0` at call time,
   `sketch_general` materializes a *temporary* representation **every call** and
   frees it. We must pre-sample (`fill_sparse(S)`) to measure apply-only cost, and
   *separately* measure the cold path.
2. **View + convert/sort** — `apply_coo_via_csc` determines the COO sort order and,
   if not already CSC-sorted, **deepcopies + re-sorts to CSC on every call**. This
   is the per-apply re-sort flagged for wide-LASO left-sketch; it must be timed in
   isolation.
3. **Kernel** — `apply_csc_jki_p11` (ColMajor) / `apply_csc_kib_1p1_rowmajor`
   (RowMajor), with the regular-CSC specialization firing only in the `jki` path
   when nnz-per-column is exactly fixed.

> Design choice: report **warm apply** (phases 2+3, `S` pre-sampled) as the primary
> number, plus a **cold** number (phases 1+2+3) and a **convert-only** number, so
> the re-sort cost is visible. This mirrors `spmm_performance`'s
> `run_split_trials` (densify vs compute) but with a sketching-specific split.

## Relationship to `spmm_performance.cc` (reuse vs. change)

**Reuse verbatim:**
- `run_trials` / `run_split_trials` (min + median over N trials), `print_row`,
  `print_table_header`, the densify+GEMM correctness oracle pattern.
- Both dense layouts (`ColMajor`, `RowMajor`) and both `op` flags — the dispatch
  selects the kernel purely from `(layout_opB, layout_C)`, exactly as documented in
  the spmm benchmark header.
- The "compute the identical logical product in both layouts and check against one
  ColMajor oracle" structure.

**Change / add:**
- Operand is a `SparseSkOp` built from a `SparseDist`, **not** `random_coo`. So the
  regular-CSC path and CountSketch are actually represented.
- New sweep axes: `major_axis ∈ {Short, Long}` and `vec_nnz ∈ {1,2,4,8}`.
- Both **sides** (left-sketch and right-sketch). Right-sketch uses the tall
  operator and, per the transpose reduction, should match the corresponding
  left-sketch RowMajor/ColMajor kernel — the benchmark lets us confirm this
  empirically (a useful cross-check, not just a measurement).
- Phase-split timing (sample / convert / kernel) described above.
- Oracle is `densify(S) + GEMM`, where `densify(S)` comes from the COO view of the
  sampled operator (reuse `coo` → dense, or build dense directly from
  `rows/cols/vals`).

## Operator shapes to sweep (the structured taxonomy)

Map each `SparseDist` onto the opt-doc taxonomy. For left-sketch, `S` is `d×m`
with `d < m` (wide); for right-sketch, `S` is `n×d` with `n > d` (tall).

| Config | dist | side | structured axis | what it isolates |
|---|---|---|---|---|
| Wide SASO | `SparseDist(d,m,k,Short)` | left | fixed `k`/col (CSC-regular) | regular-CSC fast path |
| CountSketch | `SparseDist(d,m,1,Short)` | left | 1/col | the `vec_nnz=1` special case |
| Wide LASO | `SparseDist(d,m,k,Long)` | left | ≤`k`/row, empty cols | the uncovered CSR-gather case + re-sort cost |
| Tall SASO | `SparseDist(n,d,k,Short)` | right | →CSC-regular via transpose | confirms transpose reduction == wide SASO |
| Tall LASO | `SparseDist(n,d,k,Long)` | right | ≤`k`/col after transpose | general CSC scatter (no regular) |

For each: `× {ColMajor, RowMajor} × {opS, opA NoTrans/Trans as applicable}`.

## Parameters and sweep

CLI mirrors `spmm_performance`:

```
./sketch_general_performance                                  # default sweep
./sketch_general_performance d m n vec_nnz major_axis [trials]  # single config
  major_axis: 0 = Short (SASO), 1 = Long (LASO)
```

**Default sweep** (10 trials each), two sections:

1. **Square-ish data, varying sketch size** — fix `m=n` large, sweep
   `d ∈ {m/50, m/10, m/4}`; e.g. `m=n=2000`, `d ∈ {40,200,500}`. Covers the
   common "tall data, modest embedding" regime.
2. **`vec_nnz` / `major_axis` cross** — fix one geometry (e.g. `d=200, m=n=2000`)
   and sweep `vec_nnz ∈ {1,2,4,8} × major_axis ∈ {Short, Long}`, both sides. This
   is the section that exercises CountSketch, wide LASO, and the regular path.

Also expose, via env (as in the existing benchmark): `OMP_NUM_THREADS`. Per the
perf notes, sweep `{1,4,8}` on the M3 and additionally on an Arm SVE box.

Include the **`n=1` / narrow-`n`** SpMV regime explicitly (a small extra config),
since that is the design fork for the CountSketch bucketed kernel.

## Measurement methodology

For each (shape × layout × side):

- **Pre-sample** `S` once with `fill_sparse(S)`; then time the warm
  `sketch_general` call (re-zeroing `B` each trial), reporting `{min, median}`.
- **Cold** variant: reconstruct an unsampled `SparseSkOp` each trial and time the
  full call (captures sample + convert + kernel) — report `min` only.
- **Convert-only**: time `coo_view_of_skop` + the COO→CSC sort path in isolation
  (or infer as cold − warm − sample) to expose the per-apply re-sort.
- **References**: `densify(S)+GEMM` (oracle, split densify vs GEMM like spmm), and
  optionally a `DenseSkOp` (Gaussian) sketch via `lskge3` at the same `d,m,n` for a
  "dense operator" cost anchor.
- **Correctness**: one extra call per (shape×layout×side) compared against the
  ColMajor densify+GEMM oracle, `max|diff| < 1e-10`, printed as a PASS/FAIL line.

Report a SUMMARY per config: best warm (ColMajor vs RowMajor), regular-vs-general
gap where both are reachable, and convert/re-sort as a fraction of warm apply.

## Output tables (per config)

Mirror the spmm layout — one table per (side × layout):

```
SKETCH_GENERAL  left,  ColMajor  d x n = A applied
  Operator                Min(us)  Med(us)  vs best   notes
  Wide SASO  k=4          ...                          (regular-CSC)
  CountSketch k=1         ...                          (regular, 1/col)
  Wide LASO  k=4          ...                          (general CSC + resort)
  densify(S)+GEMM         ...                          (densify .. + GEMM ..)
```

Plus the RowMajor table, the right-sketch (tall) tables, and a convert-cost line.

## File / build wiring

- Add `sketch_general_performance.cc` under
  `examples/simple-kernel-benchmarks/`.
- In `examples/CMakeLists.txt`, add an `add_executable` /
  `target_include_directories(... ${Random123_DIR})` /
  `target_link_libraries(... RandBLAS blaspp lapackpp)` block copied from the
  `spmm_performance` block (lines ~95–103).
- Headers: `<RandBLAS.hh>` is enough for `sketch_general` / `SparseSkOp` /
  `SparseDist` / `fill_sparse`; pull `RandBLAS/sparse_data/conversions.hh` only if
  densifying via the conversion helpers.

## Things to confirm while implementing

1. Whether `sketch_general` with a pre-sampled `S` still re-sorts COO→CSC on each
   call (expected yes, via `apply_coo_via_csc`) — the convert-only timing settles
   this.
2. Which kernel each (side × layout) actually lands on, to confirm the regular
   fast path is/ isn't invoked (e.g. ColMajor right-sketch → RowMajor axpy kernel,
   no regular specialization).
3. That a sampled wide SASO's COO is in CSC sort order (so no re-sort) while a
   wide LASO's is not — the structural prediction the benchmark is meant to verify.

## Measured findings (first run, M3, OMP=4)

The benchmark is implemented as `sketch_general_performance.cc` and runs clean
(all correctness checks pass). First results already correct a prediction:

- **Only CountSketch (`k=1`) is `coo-sort=CSC`** and skips the re-sort
  (convert ≈ 3µs). **Wide SASO `k≥2` is `coo-sort=None`** and re-sorts on *every*
  apply (convert ≈ 30/77/171µs for k=2/4/8 at d=40) — i.e. the per-apply re-sort
  is **not** limited to wide LASO; SASO pays it too because within-column row
  indices aren't sorted. This strengthens the case for structure-aware dispatch.
- **RowMajor is ~2× faster than ColMajor** for these operators (e.g. wide SASO
  k=4, d=40: 1144 vs 2276µs), consistent with the prior SpMM findings.
- Wide LASO has very low nnz (≤ `k·d` minus merge collisions) and is the cheapest
  case by a wide margin.
```
