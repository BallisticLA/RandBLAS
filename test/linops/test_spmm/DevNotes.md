# Sparse Matrix Multiplication Codepath Coverage Analysis

Tests are organized in three files:
- `test_spmm_coo.cc` - COO format tests
- `test_spmm_csc.cc` - CSC format tests
- `test_spmm_csr.cc` - CSR format tests

Each file has test classes for left multiplication and right multiplication. Each test class defines helper methods that call `left_apply` or `right_apply` (from `linop_common.hh`) with specific parameter combinations.

## Reduction to left-multiplication

The `right_spmm` function ([spmm_dispatch.hh:149-186](../../../RandBLAS/sparse_data/spmm_dispatch.hh#L149-L186)) reduces to `left_spmm` by:
1. Flipping `opB`: `trans_opB = (opB == NoTrans) ? Trans : NoTrans`
2. Flipping `layout`: `trans_layout = (layout == ColMajor) ? RowMajor : ColMajor`

This means `right_spmm` tests hit complementary codepaths to `left_spmm` tests. For example:
- `right_spmm(ColMajor, opA=NoTrans, opB=NoTrans)` → `left_spmm(RowMajor, opA=NoTrans, opB=Trans)`

The test files include extensive `TestRightMultiply_*` test classes that exercise right multiplication with the same test methods. 
We don't bother describing those methods in this DevNote.


## Description of left-multiplication codepaths

The `left_spmm` function dispatches to different kernels based on:
1. Matrix format: COO, CSR, or CSC
2. Memory layout (`layout` parameter): ColMajor or RowMajor
3. Dense operand transposition (`opB` parameter): NoTrans or Trans

This creates 3 × 2 × 2 = 12 combinations.

### Dispatch Details

When `opA == Op::Trans`, `left_spmm` creates a lightweight transpose view and recursively calls itself with `Op::NoTrans` ([spmm_dispatch.hh:69-73](../../../RandBLAS/sparse_data/spmm_dispatch.hh#L69-L73)):
- COO → COO (no format change)
- CSR → CSC (compressed rows become compressed columns)
- CSC → CSR (compressed columns become compressed rows)

After this transformation, all processing assumes `opA == NoTrans`.

The effective layout for reading the dense operand B (`layout_opB`) is determined from `layout` and `opB` ([spmm_dispatch.hh:96-104](../../../RandBLAS/sparse_data/spmm_dispatch.hh#L96-L104)):
- If `opB == NoTrans`: `layout_opB = layout`
- If `opB == Trans`: `layout_opB = (layout == ColMajor) ? RowMajor : ColMajor`

From there, our next steps are format-dependent.

COO format ([spmm_dispatch.hh:124-126](../../../RandBLAS/sparse_data/spmm_dispatch.hh#L124-L126)):
- Always uses `apply_coo_via_csc` (converts to CSC internally)
- 1 kernel handles all 4 (layout × opB) combinations

CSC format ([spmm_dispatch.hh:128-134](../../../RandBLAS/sparse_data/spmm_dispatch.hh#L128-L134)):
- If `layout_opB == RowMajor && layout_C == RowMajor`: `apply_csc_kib_1p1_rowmajor`
- Otherwise: `apply_csc_jki_p11`
- 2 kernels split the 4 (layout × opB) combinations:
  - `kib_rowmajor`: layout=RowMajor, opB=NoTrans
  - `jki_p11`: other 3 combinations

CSR format ([spmm_dispatch.hh:136-142](../../../RandBLAS/sparse_data/spmm_dispatch.hh#L136-L142)):
- If `layout_opB == RowMajor && layout_C == RowMajor`: `apply_csr_ikb_p1b_rowmajor`
- Otherwise: `apply_csr_jik_p11`
- 2 kernels split the 4 (layout × opB) combinations:
  - `ikb_rowmajor`: layout=RowMajor, opB=NoTrans
  - `jik_p11`: other 3 combinations


## Analysis of left-multiplication test coverage

For a non-transposed sparse matrix (after opA transformation), the 12 codepaths are:

| # | Format | layout | opB | layout_opB | layout_C | Kernel |
|---|--------|--------|-----|------------|----------|--------|
| 1 | COO | ColMajor | NoTrans | ColMajor | ColMajor | coo_via_csc |
| 2 | COO | ColMajor | Trans | RowMajor | ColMajor | coo_via_csc |
| 3 | COO | RowMajor | NoTrans | RowMajor | RowMajor | coo_via_csc |
| 4 | COO | RowMajor | Trans | ColMajor | RowMajor | coo_via_csc |
| 5 | CSC | ColMajor | NoTrans | ColMajor | ColMajor | csc_jki_p11 |
| 6 | CSC | ColMajor | Trans | RowMajor | ColMajor | csc_jki_p11 |
| 7 | CSC | RowMajor | NoTrans | RowMajor | RowMajor | csc_kib_1p1_rowmajor |
| 8 | CSC | RowMajor | Trans | ColMajor | RowMajor | csc_jki_p11 |
| 9 | CSR | ColMajor | NoTrans | ColMajor | ColMajor | csr_jik_p11 |
| 10 | CSR | ColMajor | Trans | RowMajor | ColMajor | csr_jik_p11 |
| 11 | CSR | RowMajor | NoTrans | RowMajor | RowMajor | csr_ikb_p1b_rowmajor |
| 12 | CSR | RowMajor | Trans | ColMajor | RowMajor | csr_jik_p11 |


There are five types of tests called for each sparse matrix format (and both memory layouts): `multiply_eye`, `alpha_beta`, `transpose_self`, `transpose_other`, `submatrix_other`, and `submatrix_self` (COO-only).

Left-multiplication in COO Format (test_spmm_coo.cc)

- Paths 1-2: `multiply_eye`, `alpha_beta`, `submatrix_other` (ColMajor, opB=NoTrans)
- Paths 3-4: `multiply_eye`, `alpha_beta`, `submatrix_other` (RowMajor, opB=NoTrans)
- Paths 1-4: `transpose_other` tests opB=Trans for both layouts
- Paths 1-4: `transpose_self` tests opA=Trans (COO→COO) for both layouts

Left-multiplication in CSC Format (test_spmm_csc.cc)

- Path 5: `multiply_eye`, `alpha_beta` (ColMajor, opB=NoTrans)
- Path 6: `transpose_other` (ColMajor, opB=Trans)
- Path 7: `multiply_eye`, `alpha_beta` (RowMajor, opB=NoTrans) → kib_rowmajor kernel
- Path 8: `transpose_other` (RowMajor, opB=Trans)
- Paths 9-12: `transpose_self` with opA=Trans converts CSC to CSR view
  - This exercises the CSR kernels

Left-multiplication in CSR Format (test_spmm_csr.cc)

- Path 9: `multiply_eye`, `alpha_beta` (ColMajor, opB=NoTrans)
- Path 10: `transpose_other` (ColMajor, opB=Trans)
- Path 11: `multiply_eye`, `alpha_beta` (RowMajor, opB=NoTrans) → ikb_rowmajor kernel
- Path 12: `transpose_other` (RowMajor, opB=Trans)
- Paths 5-8: `transpose_self` with opA=Trans converts CSR to CSC view
  - This exercises the CSC kernels

For completeness, here's a grouping of tests based on the kernel they hit.

| Kernel | Direct Tests | Via Transformation |
|--------|--------------|-------------------|
| `apply_coo_via_csc` | All COO tests | CSR/CSC transpose_self |
| `apply_csc_jki_p11` | CSC ColMajor (both opB), CSC RowMajor + opB=Trans | CSR transpose_self |
| `apply_csc_kib_1p1_rowmajor` | CSC RowMajor + opB=NoTrans | CSR transpose_self + RowMajor |
| `apply_csr_jik_p11` | CSR ColMajor (both opB), CSR RowMajor + opB=Trans | CSC transpose_self |
| `apply_csr_ikb_p1b_rowmajor` | CSR RowMajor + opB=NoTrans | CSC transpose_self + RowMajor |
