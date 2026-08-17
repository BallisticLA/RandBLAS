# RandBLAS Style Review and Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review the full RandBLAS repository, document the style that its first-party files actually establish, identify files that materially deviate from their peers, and publish a concise contributor style guide.

**Architecture:** Treat the review as an evidence pipeline: inventory the repository, divide files into comparable strata, measure dominant conventions within each stratum, rank and manually classify outliers, then promote only well-supported conventions into the guide. Keep the transient audit evidence and outlier register in `STYLE_AUDIT.md`; keep normative guidance in `STYLE_GUIDE.md`; add a short link from `README.md` so contributors can find the guide.

**Tech Stack:** Git, ripgrep, awk, Markdown, C++20/header-only library conventions, GoogleTest, CMake, Python, and Sphinx/reStructuredText.

**Spec:** Direct user request in the Codex task dated 2026-08-16; project constraints are in `AGENTS.md`.

## Global Constraints

- Review the repository at one recorded commit SHA so every count and conclusion is reproducible.
- Analyze all tracked first-party text files, but compare each file only with appropriate peers: library headers, tests, examples/benchmarks, CMake, scripts/workflows, and documentation.
- Exclude data fixtures, generated files, vendored code, and machine-readable lock/config data from dominant-style calculations; list every exclusion and its reason in the audit.
- Use a convention as a descriptive project norm only when it appears in at least five eligible files and at least 70% of eligible occurrences. Label 90% or greater as strong consensus.
- When the repository is mixed, distinguish an explicitly chosen forward-looking recommendation from an observed consensus and record the rationale.
- Do not reformat or clean up source files while performing this review. Outlier remediation is a separate follow-up project.
- Do not add `.clang-format`, `.clang-tidy`, or CI enforcement in this project; the requested deliverables are the audit and style guide.
- Preserve the existing untracked `.claude/` directory and all unrelated user changes.
- No build is required for documentation-only changes. If execution touches C++ or CMake unexpectedly, stop and use the workspace's Spack environment via `source sourceme.sh` before building or testing.

---

## Planned File Structure

- Create: `STYLE_AUDIT.md` — snapshot metadata, corpus classification, measured convention tables, outlier method, ranked outlier register, and unresolved decisions.
- Create: `STYLE_GUIDE.md` — normative contributor guidance derived from the audit, with short positive and negative examples.
- Modify: `README.md` — add one contributor-facing link to `STYLE_GUIDE.md` in the existing documentation/development area; do not otherwise reorganize the README.
- Create: `docs/superpowers/plans/2026-08-16-randblas-style-guide-review.md` — this execution plan.

### Task 1: Freeze the Review Snapshot and Classify the Corpus

**Files:**
- Create: `STYLE_AUDIT.md`

**Interfaces:**
- Consumes: tracked files and history at the current RandBLAS `HEAD`; repository rules in `AGENTS.md`.
- Produces: a stable corpus manifest and stratum definitions used by all later scoring and prose.

- [ ] **Step 1: Record repository state without changing it**

Run:

```bash
git rev-parse HEAD
git status --short
git log -5 --oneline
```

Record the full SHA, review date, branch name, and pre-existing worktree changes at the top of `STYLE_AUDIT.md`. Explicitly note that `.claude/` predated the audit and is out of scope.

- [ ] **Step 2: Inventory tracked files by extension and top-level area**

Run:

```bash
git ls-files | awk '
function ext(path, base,n,a) {
    n=split(path,a,"/"); base=a[n]
    if (base !~ /\./) return "[none]"
    sub(/^.*\./,"",base); return "." base
}
{ by_ext[ext($0)]++; split($0,p,"/"); by_root[p[1]]++ }
END {
    print "Extensions:"
    for (key in by_ext) print by_ext[key], key
    print "Top-level areas:"
    for (key in by_root) print by_root[key], key
}'
```

Sort the two result groups numerically before transcribing them into the audit. The initial review found 32 `.hh` and 37 `.cc` files; investigate any materially different count rather than silently using stale numbers.

- [ ] **Step 3: Assign every relevant text file to one comparison stratum**

Use these exact strata in `STYLE_AUDIT.md`:

1. Public/core headers: `RandBLAS.hh`, `RandBLAS/*.hh`.
2. Sparse and testing support headers: `RandBLAS/sparse_data/*.hh`, `RandBLAS/testing/*.hh`.
3. Tests: `test/**/*.cc`, `test/**/*.hh`.
4. Examples and benchmarks: `examples/**/*.cc`.
5. Build/configuration: every tracked `CMakeLists.txt`, `CMake/*.cmake`, `*.cmake.in`, and `*.h.in`.
6. Documentation: root `*.md`, `RandBLAS/**/DevNotes.md`, `test/**/DevNotes.md`, and `rtd/**/*.{rst,md}`.
7. Python and web-theme sources: tracked `*.py`, `*.js`, and `*.css`.
8. Automation/install scripts: tracked `*.sh`, `*.ps1`, `*.yml`, and `*.yaml` under installer, Docker, or workflow paths.

Make an exclusions table with columns `Path or glob`, `Reason`, and `Compared elsewhere?`. At minimum, exclude numerical fixtures such as `*.mm`, known-answer vector data such as `r123_kat_vectors.txt`, licenses, images, and XML/Doxygen configuration from prose or C++ style calculations. Do not label an excluded file an outlier merely because its format differs.

- [ ] **Step 4: Record representative manual samples**

In addition to the full automated scan, select these anchors for close reading:

- Core API: `RandBLAS/base.hh`, `RandBLAS/dense_skops.hh`, `RandBLAS/skge.hh`.
- Sparse internals: `RandBLAS/sparse_data/csr_spmm_impl.hh`, `RandBLAS/sparse_data/spmm_dispatch.hh`, `RandBLAS/sparse_data/csr_matrix.hh`.
- Test code: `test/datastructures/test_denseskop.cc`, `test/linops/test_lskge3.cc`, `test/linops/test_spmm/test_spmm_csr.cc`.
- Examples: `examples/total-least-squares/tls_dense_skop.cc`, `examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc`, `examples/simple-kernel-benchmarks/spmm_performance.cc`.
- Non-C++: `CMakeLists.txt`, `CMake/rb_config.cmake`, `rtd/source/conf.py`, `rtd/source/tutorial/gemm.rst`, `README.md`.

Explain that anchors illustrate style but do not override full-corpus counts.

- [ ] **Step 5: Verify corpus coverage**

Run `git ls-files` and account for each first-party text suffix in either a stratum or the exclusions table. Add a `Coverage` summary stating the total files classified, total excluded, and any ambiguous files with a concrete disposition.

- [ ] **Step 6: Commit the inventory checkpoint**

```bash
git add STYLE_AUDIT.md
git commit -m "docs: inventory sources for style review"
```

### Task 2: Measure Dominant Conventions Within Each Stratum

**Files:**
- Modify: `STYLE_AUDIT.md`

**Interfaces:**
- Consumes: the frozen corpus and stratum assignments from Task 1.
- Produces: per-convention evidence with eligible counts, dominant shares, confidence labels, and representative paths.

- [ ] **Step 1: Scan universal hygiene conventions**

Run:

```bash
git ls-files -z '*.hh' '*.cc' '*.cmake' '*.md' '*.rst' '*.py' '*.sh' '*.ps1' '*.yml' '*.yaml' | xargs -0 rg -n '[ \t]+$'
while IFS= read -r header_file; do rg -q '^#pragma once$' "$header_file" || echo "$header_file"; done < <(git ls-files '*.hh')
while IFS= read -r source_file; do rg -q '^// Copyright,' "$source_file" || echo "$source_file"; done < <(git ls-files '*.hh' '*.cc')
git ls-files '*.hh' '*.cc' | xargs awk 'length($0) > 100 { n[FILENAME]++ } END { for (f in n) print n[f], f }'
```

Record trailing-whitespace incidence, header protection, license-header coverage, and long-line distribution. Treat URLs, long strings, macro continuations, and mathematical expressions as separately counted exceptions rather than evidence for a broad line-length rule.

- [ ] **Step 2: Profile C++ formatting and organization**

Count eligible occurrences and variants for:

- four-space indentation and absence of tabs;
- `template <...>` versus `template<...>`;
- same-line opening braces versus next-line braces for namespaces, types, functions, and control flow;
- mandatory braces versus single-statement unbraced loops/conditionals;
- spaces around inheritance colons, operators, commas, and template arguments;
- pointer/reference attachment (`T *p`, `T* p`, `T &x`, `T& x`), keeping declarations separate from expressions;
- project includes versus third-party/system includes, quote/angle form, grouping, alphabetical order, and duplicates;
- preprocessor directive indentation, especially OpenMP and local matrix-index macros;
- namespace declaration and closing-comment patterns;
- redundant semicolons after inline member function definitions.

Use `rg -n` searches that print path and line number, then summarize counts in the audit rather than pasting raw output. Useful seed searches are:

```bash
rg -n '^\s*template<' RandBLAS test examples -g '*.{hh,cc}'
rg -n '^\s*(class|struct)\s+[^:{]+\s*:' RandBLAS test examples -g '*.{hh,cc}'
rg -n '^\s*#\s+(include|define|pragma)' RandBLAS test examples -g '*.{hh,cc}'
rg -n '#include\s*[<"]' RandBLAS test examples -g '*.{hh,cc}'
rg -n '^\s*namespace\s+[^ {]+$' RandBLAS test examples -g '*.{hh,cc}'
rg -n '^\s*};\s*$' RandBLAS test examples -g '*.{hh,cc}'
```

- [ ] **Step 3: Profile C++ naming and API conventions**

Build an evidence table for:

- `RandBLAS` and nested namespace naming;
- PascalCase public types and enum types;
- snake_case functions, variables, fields, and filenames;
- uppercase macro names and the special local matrix-index macro pattern;
- template parameter naming, separating scalar types (`T`), RNG/operator types, and integer/index types;
- BLAS++ enum usage, layout/op/side argument ordering, and `int64_t`/signed-index conventions;
- public API parameter wrapping and one-parameter-per-line signatures;
- C++20 concepts and compatibility preprocessor fallbacks.

For every proposed naming rule, include at least three conforming examples from at least two files and list counterexamples. Do not infer a public API rule solely from test helpers or examples.

- [ ] **Step 4: Profile documentation and comment conventions**

Measure and compare:

- file-level `/// @file` usage;
- `///` versus `/** ... */` Doxygen comments;
- `@tparam`, `@param[in]`, and `@returns` spelling and placement;
- comments that explain invariants or performance rationale versus comments that restate code;
- Markdown and RST heading depth, code-block style, mathematical notation, and link form;
- DevNotes' role compared with public ReadTheDocs content.

Record misspellings or malformed markup as quality findings, not style norms.

- [ ] **Step 5: Profile tests, examples, CMake, Python, and automation separately**

Document the modal patterns for GoogleTest fixture and test names, typed-test organization, assertion choice, helper placement, and test-data naming. Separately document example/benchmark CLI and timing conventions, CMake command/variable naming and indentation, Python's indentation/import/docstring style, and workflow/script naming. Require the same five-file/70% threshold where the stratum is large enough; for smaller strata, state that guidance is a maintainer recommendation supported by all available examples rather than a statistical consensus.

- [ ] **Step 6: Label confidence and resolve contradictions**

For each convention row, calculate `dominant occurrences / eligible occurrences` and assign:

- `Strong consensus`: at least 90% and at least five eligible files.
- `Working consensus`: at least 70% and at least five eligible files.
- `Mixed`: below 70%, or fewer than five eligible files.

For mixed rows, choose one of three explicit dispositions: omit from the guide, document stratum-specific variants, or recommend a forward-looking rule with a short maintainability rationale. Never present a mixed observation as established house style.

- [ ] **Step 7: Commit the convention profile**

```bash
git add STYLE_AUDIT.md
git commit -m "docs: profile RandBLAS style conventions"
```

### Task 3: Rank and Manually Classify Outlier Files

**Files:**
- Modify: `STYLE_AUDIT.md`

**Interfaces:**
- Consumes: eligible convention rows and confidence values from Task 2.
- Produces: a reviewed outlier register that distinguishes cleanup candidates from intentional exceptions and feeds the guide's exception policy.

- [ ] **Step 1: Calculate a within-stratum deviation score**

For every eligible file and convention, calculate a deviation rate:

```text
deviation_rate = nonconforming eligible occurrences / all eligible occurrences
convention_weight = dominant convention share
file_score = sum(deviation_rate * convention_weight) / sum(applicable convention_weight)
```

Do not compare raw violation counts because large files would be penalized simply for size. Rank files only against peers in the same stratum.

- [ ] **Step 2: Select objective outlier candidates**

Flag a file for manual review when either condition holds:

1. It is in the top 10% of scores within a stratum and its score is at least 0.20.
2. It violates at least two strong-consensus structural conventions, regardless of total score.

Structural conventions include license/header placement, duplicate protection or includes, include grouping, namespace form, and pervasive indentation or brace style. A single typo is a defect finding but does not by itself make the entire file a style outlier.

- [ ] **Step 3: Start with the observed seed candidates, then accept or reject them using the scoring rule**

The initial inspection found these concrete candidates; they are not final classifications:

| Candidate | Initial evidence to verify | Likely classification to test |
|---|---|---|
| `RandBLAS/sparse_data/csr_trsm_impl.hh` | `#pragma once` appears both before and after the license header | Cleanup candidate |
| `RandBLAS/base.hh` | duplicate `<iostream>` include; one occurrence is written `#include<iostream>` | Cleanup candidate |
| `RandBLAS/util.hh` | project headers use angle brackets while most core headers use quotes; local matrix-index macros are frequent | Mixed: cleanup candidate or justified local convention |
| `RandBLAS/random_gen.hh` | the `r123ext` block uses next-line braces while much of the library uses same-line braces | Intentional inherited style or cleanup candidate |
| `test/datastructures/test_denseskop.cc` | repeated `template<...>` forms and `public::testing::Test` without normal spacing | Cleanup candidate |
| `test/basic_rng/test_r123.cc` | unusually macro-heavy known-answer test infrastructure | Intentional legacy/upstream-shaped exception |
| `examples/sparse-low-rank-approx/qrcp_matrixmarket.cc` | local timing macros and application-specific formatting | Intentional example-local exception or cleanup candidate |
| `examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc` | indented matrix-index macros and application-specific formatting | Intentional example-local exception or cleanup candidate |
| `examples/sparse-low-rank-approx/svd_matrixmarket.cc` | local timing macros and application-specific formatting | Intentional example-local exception or cleanup candidate |

Add any higher-scoring files found by the full scan; remove seeds that do not meet either objective threshold, explaining the rejection in one sentence.

- [ ] **Step 4: Review history and provenance for every candidate**

Run:

```bash
git log --follow --format='%h %ad %s' --date=short -- RandBLAS/sparse_data/csr_trsm_impl.hh
git blame -w -- RandBLAS/sparse_data/csr_trsm_impl.hh
```

Repeat both commands with each other candidate's exact quoted path. Record the commands and findings under that candidate's audit entry.

Use history only to explain provenance, not to excuse accidental inconsistency. Look for imported upstream test code, generated output, platform-specific constraints, intentionally optimized kernels, or old files that predate the current modal style.

- [ ] **Step 5: Publish the final outlier register**

Add a table to `STYLE_AUDIT.md` with columns:

`Rank`, `File`, `Stratum`, `Score`, `Strong deviations`, `Provenance`, `Classification`, `Recommended action`.

The only allowed classifications are:

- `Cleanup candidate`: should follow the guide in a future focused patch.
- `Intentional exception`: local form is justified; the guide must state the narrow exception.
- `Mixed evidence`: maintainers must choose before enforcement.
- `Not an outlier`: seed rejected by the objective thresholds.

Keep remediation recommendations narrow. Do not propose bulk formatting in this audit.

- [ ] **Step 6: Commit the reviewed outlier register**

```bash
git add STYLE_AUDIT.md
git commit -m "docs: identify RandBLAS style outliers"
```

### Task 4: Draft the Normative Style Guide

**Files:**
- Create: `STYLE_GUIDE.md`

**Interfaces:**
- Consumes: convention evidence, confidence labels, and intentional exceptions from `STYLE_AUDIT.md`.
- Produces: contributor-facing rules that can be applied without re-reading the audit.

- [ ] **Step 1: Create the guide outline**

Use these sections in this order:

1. Purpose and scope.
2. Guiding principles: readability, numerical correctness, reproducibility across OpenMP thread counts, and performance awareness.
3. File layout: license, `#pragma once`, includes, namespaces, declarations, definitions.
4. C++ formatting: indentation, braces, spacing, line wrapping, pointers/references, templates, preprocessor directives.
5. Naming: namespaces, public types, functions, variables, template parameters, macros, tests, and files.
6. Public APIs and templates: C++20 concepts, BLAS++ types, dimensions/strides, validation, and overload consistency.
7. Includes and dependencies: quote/angle policy, grouping, self-sufficiency, and duplicate avoidance.
8. Comments and documentation: Doxygen, mathematical notation, invariants, DevNotes, and ReadTheDocs.
9. Parallel and performance-sensitive code: OpenMP, thread-count-independent random generation, dispatch paths, and benchmarking expectations.
10. Tests: GoogleTest naming, abstraction-level placement, deterministic/statistical coverage, and sparse dispatch coverage.
11. Examples, CMake, Python, and automation.
12. Intentional exceptions and how to document a new one.
13. Contributor checklist.

- [ ] **Step 2: Write only evidence-backed rules**

For every rule, either cite representative repository paths inline or link to the corresponding convention row in `STYLE_AUDIT.md`. Use `must` for correctness/build requirements already mandated by `AGENTS.md`, `should` for strong or working style consensus, and `may` for documented exceptions. Clearly prefix forward-looking choices with `Recommendation:`.

- [ ] **Step 3: Add compact examples**

Include positive/negative examples for the conventions most likely to be misread: include grouping, multi-line function signatures, template spacing, pointer/reference spacing, braces, Doxygen parameters, GoogleTest fixtures, and OpenMP preprocessor placement. Keep each example focused on one rule and ensure it does not contradict real public API constraints.

- [ ] **Step 4: Encode outlier policy without naming accidental code as precedent**

State that existing deviations do not establish a second style. Name only files classified as intentional exceptions, explain the narrow reason, and link to their audit entry. Leave cleanup candidates in `STYLE_AUDIT.md`, not in the normative body of the guide.

- [ ] **Step 5: Add the contributor checklist**

The checklist must cover formatting, include self-sufficiency, API consistency, Doxygen updates, RNG thread independence, all affected sparse dispatch paths, focused tests, benchmarks for performance changes, and full `ctest` execution through the workspace's Spack environment after code changes.

- [ ] **Step 6: Cross-check the guide against repository constraints**

Compare the guide line by line with `AGENTS.md`, `RandBLAS/DevNotes.md`, `RandBLAS/sparse_data/DevNotes.md`, and `test/DevNotes.md`. Resolve contradictions in favor of those project instructions and record any surprising resolution in the audit.

- [ ] **Step 7: Commit the draft guide**

```bash
git add STYLE_GUIDE.md STYLE_AUDIT.md
git commit -m "docs: add evidence-based RandBLAS style guide"
```

### Task 5: Validate Traceability, Usability, and Discoverability

**Files:**
- Modify: `STYLE_GUIDE.md`
- Modify: `STYLE_AUDIT.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the completed audit and guide.
- Produces: a self-consistent, discoverable documentation set with every norm traceable to evidence or an explicit recommendation.

- [ ] **Step 1: Build a rule-to-evidence traceability table**

At the end of `STYLE_AUDIT.md`, add columns `Guide section`, `Rule summary`, `Evidence row`, `Confidence`, and `Exception`. Every normative `must` or `should` in `STYLE_GUIDE.md` must have a row. Remove or relabel any rule that lacks evidence or a project-instruction source.

- [ ] **Step 2: Dry-run the guide as a review rubric**

Apply it without editing source to these pairs:

- Core headers: `RandBLAS/dense_skops.hh` and candidate `RandBLAS/base.hh`.
- Sparse headers: `RandBLAS/sparse_data/csr_spmm_impl.hh` and candidate `RandBLAS/sparse_data/csr_trsm_impl.hh`.
- Tests: `test/linops/test_lskge3.cc` and candidate `test/datastructures/test_denseskop.cc`.
- Examples: `examples/total-least-squares/tls_dense_skop.cc` and the highest-ranked sparse-low-rank example candidate.
- Non-C++: `CMake/rb_config.cmake` and `rtd/source/conf.py`.

For each file, record whether two reviewers following the written rule would reach the same result. Rewrite ambiguous rules until the expected judgment is explicit.

- [ ] **Step 3: Add a README link**

Add one sentence in the existing development or documentation section of `README.md` linking to `STYLE_GUIDE.md`. If neither section exists, add a two-line `Contributing` subsection near the installation/documentation links rather than creating a large new contributor guide.

- [ ] **Step 4: Run documentation-focused verification**

Run:

```bash
git diff --check
rg -n 'TBD|TODO|implement later|fill in details|similar to Task' STYLE_GUIDE.md STYLE_AUDIT.md
rg -n 'STYLE_GUIDE.md' README.md
test -f STYLE_GUIDE.md && test -f STYLE_AUDIT.md
```

Expected results: `git diff --check` is silent; the placeholder scan is silent; the README search reports the new link; both file tests succeed.

- [ ] **Step 5: Review the final diff for scope**

Run:

```bash
git status --short
git diff --stat HEAD~4
git diff HEAD~4 -- README.md STYLE_GUIDE.md STYLE_AUDIT.md
```

Confirm that no C++, CMake, test, generated, or user-owned file was reformatted or changed. If code did change, revert only the audit-created code edits and leave unrelated user changes untouched.

- [ ] **Step 6: Commit integration changes**

```bash
git add README.md STYLE_GUIDE.md STYLE_AUDIT.md
git commit -m "docs: link and validate RandBLAS style guidance"
```

## Completion Criteria

- `STYLE_AUDIT.md` identifies the reviewed commit, accounts for the corpus, reports convention frequencies, and contains a ranked, manually classified outlier register.
- `STYLE_GUIDE.md` clearly separates observed consensus, correctness mandates, forward-looking recommendations, and intentional exceptions.
- Every normative rule is traceable to repository evidence or an existing project instruction.
- Outliers are identified relative to peers, with data/generated/vendored content protected from false classification.
- `README.md` links to the guide.
- Documentation verification passes, and the task contains no source-code reformatting or cleanup.
