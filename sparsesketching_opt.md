# Optimizing SPMM kernels for sparse sketching operators

A proposal for new structure-aware SPMM kernels in RandBLAS, plus a recommendation
to gather targeted performance data before hand-tuning.

> **Revision note.** An earlier draft listed *tall SASO (right-sketching)* as a
> motivating case for a new regular-CSR kernel. That was wrong: free operator
> transposition plus the `right_spmm`→`left_spmm` reduction already turn that case
> into the existing wide-SASO / CSC-regular kernel. This draft reframes the
> taxonomy around the operator's *structured axis* (row vs column), which is the
> distinction that survives transposition.

## How sketching operators reach the kernels today

Every `SparseSkOp` is stored as **COO** (`rows, cols, vals`). Sketching
(`lskges`/`rskges`) takes a `coo_view_of_skop` and calls `left_spmm`/`right_spmm`.

Two reductions happen *before* any kernel runs, and they are central to this
analysis:

- **`right_spmm` reduces to `left_spmm`** by transposing the operation and
  **flipping the dense layout** (`trans_layout`: ColMajor↔RowMajor), so the sparse
  operand always lands as the left operand of `left_spmm`.
- **`left_spmm` resolves `opA=Trans` for free**: it takes a lightweight transpose
  view of the operator (for COO, just swap the `rows`/`cols` arrays) and recurses
  as `NoTrans`. Transposing a sketching operator costs nothing.

After those reductions, a COO operator goes to `apply_coo_via_csc`, which:

1. Determines the COO sort order by scanning (set in the view constructor via
   `coo_arrays_determine_sort`).
2. If not already CSC-sorted, **deepcopies and re-sorts to CSC on every call**.
3. Calls `apply_csc_kib_1p1_rowmajor` (when `layout_opB == layout_C == RowMajor`)
   or `apply_csc_jki_p11` (otherwise).

The only place structure is exploited is `apply_csc_jki_p11`: it scans `colptr`
to detect fixed-nnz-per-column and, if so, calls `apply_regular_csc_to_vector_ki`.
Note this detection lives **only** in the `jki` path — the RowMajor axpy kernel
`apply_csc_kib_1p1_rowmajor` has no regular specialization.

## The right way to think about it: the *structured axis*, not tall vs wide

Because transposition is free and `right_spmm` already flips layout, **"tall vs
wide" is not an independent design axis** — it is absorbed by free transpose plus
a dense-layout flip. What survives transposition is whether the operator is
**column-structured** (feeds the CSC *scatter* kernel) or **row-structured**
(wants a CSR *gather* kernel). A transpose swaps a row-structured operator into a
column-structured one, but it simultaneously swaps the roles/layouts of the dense
operands — so it only helps when the dispatch is *already* transposing for another
reason.

Re-cast the four shapes accordingly:

| Operator | Usage | Structured axis after reductions | Reaches which kernel |
|---|---|---|---|
| **Wide SASO** (d×m) | left-sketch | exactly `vec_nnz` per **column** (CSC-regular) | ✅ `apply_regular_csc...` |
| **Tall SASO** (m×d) | right-sketch | `right_spmm` transposes → **column-regular** (= wide SASO) | ✅ same CSC-regular machinery — **no new kernel needed** |
| **CountSketch** | left-sketch | column-regular with `vec_nnz = 1` | ✅ regular-CSC, but very under-specialized |
| **Wide LASO** (d×m) | left-sketch | **≤** `vec_nnz` per **row**, mostly-empty cols (CSR) | ❌ no auto-transpose → general CSC scatter over all `m` cols |
| **Tall LASO** (m×d) | right-sketch | transposes → **≤** per column (CSC, not exact) | ⚠️ general CSC scatter; regular detector can't fire (counts vary) |

The two takeaways:

1. **Tall SASO needs no new *kernel*.** Its row structure is converted to the
   existing column-regular kernel by the transpose reduction. (Measurement caveat:
   the COO view's sort flag comes back `None` for `vec_nnz ≥ 2` — within-vector
   indices aren't fully ordered — so `apply_coo_via_csc` still re-sorts on every
   apply; only CountSketch, `vec_nnz = 1`, is detected as CSC-sorted. See the
   benchmark findings in `sparsesketching_bench_plan.md`. This is a dispatch cost,
   not a missing kernel.)
2. The genuinely uncovered case is the **row-structured operator that reaches a
   kernel *without* an auto-transpose** — i.e., **wide LASO in left-sketching**.
   There the transpose trick doesn't help (it would just swap the dense roles
   back), so we land on the general CSC scatter, which iterates over all `m`
   columns (most empty) and sees no regularity.

A caveat that matters for *invocation* (not just structure): the regular-CSC fast
path lives only in `apply_csc_jki_p11`. A common **ColMajor right-sketch with a
tall SASO** flips to RowMajor and routes to `apply_csc_kib_1p1_rowmajor` (the
axpy kernel), which does **not** specialize on regularity. So "structure covered"
≠ "fast path invoked" — which kernel fires is layout-dependent.

## Proposed kernels

**1. CountSketch kernels (`vec_nnz == 1`)** — highest value and most
self-contained. A wide CountSketch maps each column `c` to one row `h(c)` with
sign `s(c)`: `C[h(c),:] += s(c)·B[c,:]`. Today this runs through regular-CSC with
`col_nnz=1` — correct but carrying full loop/index overhead for a length-1 inner
loop. Dedicated variants:

- **Bucketed / CSR-by-output-row form**: invert the hash once (group source
  columns by target row). Each of the `d` output rows is then an independent
  signed sum of B-rows → embarrassingly parallel over `d` with **no races and no
  per-thread rescan** (unlike `apply_csc_kib_1p1_rowmajor`, which loops all `m`
  per thread and filters by row range). This is the right structure when `n` is
  small (SpMV / few RHS), where parallel-over-`n` starves threads.
- Collapse the length-1 inner loop entirely.

**2. Bounded-CSR gather kernel for wide LASO (left-sketch).** This is the case the
transpose reduction does *not* rescue. Build a CSR/gather kernel that iterates
only over the `d` nonempty rows (≤ `vec_nnz` each) instead of scattering over all
`m` columns. Realizing this means routing the operator COO→**CSR** instead of CSC
— which *also* eliminates the per-apply re-sort, since a wide LASO's natural COO
order is already CSR. A `≤`-aware variant (explicit short per-row counts, or
padding) handles the merge-induced count variation that keeps LASOs out of the
*exact* regular path.

**3. Implicit ±1-value kernels.** SASO values are exactly ±1; the isometry scale
is applied separately via `alpha`. If the prior profiling's memory-bound
hypothesis holds, **not loading the `vals` array** (encode sign in the index, or
split into +/− index lists and do two sign-free accumulations) cuts the
sparse-operand traffic by ~⅓. Machine-independent in the regime the M3 profiling
suggested, and it composes with kernels 1–2.

**4. Close the regular-path layout gap.** Add a regular-aware specialization to the
RowMajor axpy kernel (`apply_csc_kib_1p1_rowmajor`), and/or a dedicated ColMajor
row-axpy ("BLAS-3-flavored") path. Per the perf notes, ColMajor is the BLAS
default yet routes to the slow scalar scatter/gather; RowMajor gets the fast
axpy kernels and is up to ~1.9× faster.
A structured ±1 operator makes a well-vectorized ColMajor kernel far easier than
for a general sparse matrix, and it ensures the regular fast path is reachable
regardless of which layout the reductions land in.

## Cross-cutting infrastructure (needed for the above to pay off)

- **Structure-aware dispatch carrying the operator's metadata.** The skop already
  knows `major_axis`, its shape, and `vec_nnz`, so it knows its structured axis
  and whether counts are exact or bounded. Thread that through `lskges`/`rskges`
  and **choose the canonical orientation** (transpose for free so the operator is
  column-structured whenever that lands on the CSC-regular kernel; route to CSR
  when it's genuinely row-structured and not auto-transposed). This replaces the
  re-derive-by-scanning-`colptr` heuristic and removes the wrong-format re-sort.
- **A `≤`-regular variant** so LASO operators get a fast path despite
  merge-induced count variation.

## Recommendation: gather targeted data first

I'd **benchmark before writing kernels**, for concrete reasons grounded in the
prior investigation:

1. **The existing `spmm_performance` benchmark already covers both layouts and
   sides, but only on generic uniform-random matrices — not these operator
   shapes.** It drives `left_spmm` in ColMajor *and* RowMajor (plus `op(B)=Trans`)
   across CSR/CSC/COO, and its header documents that this reaches what `left_spmm`
   and `right_spmm` together hit via the transpose reduction. But it builds
   matrices with `random_coo` (uniform density), so it never produces the
   *structured* operators we care about: fixed-nnz-per-column/row, ±1 values, or
   mostly-empty columns. Consequently it **never exercises the regular-CSC fast
   path** (random density → variable nnz per column → the detector in
   `apply_csc_jki_p11` fails) and never goes through `sketch_general`. No baseline
   isolates wide-LASO, CountSketch, or the regular-structure kernels — that's the
   gap to close.
2. **Which kernel actually fires is the product of (operator shape × side ×
   layout)** — e.g. tall-SASO right-sketch reaches the regular fast path only when
   the original layout is RowMajor (ColMajor right-sketch flips to RowMajor and
   lands on the axpy kernel, which has no regular specialization). The benchmark
   already crosses layout and side; what it must add is the *structured* operators,
   so the regular path and the ±1/mostly-empty structure are actually represented.
3. **The `n`-regime decides the kernel design.** Wide `n` (parallel-over-`n` is
   fine) vs narrow `n`/SpMV (needs operator-dimension parallelism, which reshapes
   kernel 1) is the biggest design fork. Measure the shapes RandLAPACK actually
   drives.
4. **The memory-bound hypothesis is inferred from stack shape, not counters** (SIP
   gates perf counters on the M3). If it holds, kernel 3 (drop `vals`) and
   index-width (int32 vs int64) dominate arithmetic restructuring — reordering the
   priority list.
5. **The per-apply re-sort is a real cost worth quantifying** — and, per the
   benchmark, it hits *every* SASO with `vec_nnz ≥ 2` (not just wide LASO), since
   only `vec_nnz = 1` is detected as CSC-sorted. It scales with nnz and argues for
   fixing dispatch (mark/skip the sort) before chasing kernel-level gains.

Concretely: extend the harness to sweep {wide SASO, CountSketch, wide LASO} ×
{left, right} × {RowMajor, ColMajor} × a range of `n` (including `n=1` SpMV) ×
`vec_nnz ∈ {1,2,4,8}`, on both the M3 and an Arm SVE box (NEON/SVE is a
first-class target). That isolates which kernel actually moves the needle before
we hand-tune any of them.

## Suggested next step

Either:

- **Extend the benchmark** to cover these shapes/sides/layouts and produce the
  baseline, or
- **Prototype one kernel first** — CountSketch is the highest-leverage and most
  self-contained.
