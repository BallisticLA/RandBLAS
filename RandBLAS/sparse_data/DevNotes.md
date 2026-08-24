# Developer Notes for RandBLAS' sparse matrix functionality

RandBLAS provides abstractions for CSC, CSR, and COO-format sparse matrices.
The following functions use these abstractions:

 * ``left_spmm``, which computes a product of a sparse matrix and a dense matrix when the sparse matrix
    is the left operand. This function is GEMM-like, in that it allows offsets and transposition flags
    for either argument.
 * ``right_spmm``, which is analogous to ``left_spmm`` when the sparse matrix is the right operand.
 * ``sketch_general``, when called with a SparseSkOp object.
 * ``sketch_sparse``, when called with a DenseSkOp object.

Each of those functions is merely a _dispatcher_ of other (lower level) functions. See below for details on
how the dispatching works.

## Left_spmm and right_spmm

These functions are implemented in ``RandBLAS/sparse_data/spmm_dispatch.hh``.

``right_spmm`` is implemented by falling back on ``left_spmm`` with transformed
values for ``opA, opB`` and ``layout``.
Here's what happens if ``left_spmm`` is called with a sparse matrix ``A``, a dense input matrix ``B``, and a dense output matrix ``C``.

 1. If needed, transposition of ``A`` is resolved by creating a lightweight object for the transpose
    called ``At``. This object is just a tool for us to change how we interpret the buffers that underlie ``A``.
      * If ``A`` is COO, then ``At`` will also be COO.
      * If ``A`` is CSR, then ``At`` will be CSC.
      * If ``A`` is CSC, then ``At`` will be CSR.
    
    We make a recursive call to ``left_spmm`` once we have our hands on ``At``, so
    the rest of ``left_spmm``'s logic only needs to handle un-transposed ``A``.

 2. A memory layout is determined for how we'll read ``B`` in the low-level 
    sparse matrix multiplication kernels.
      * If ``B`` is un-transposed then we'll use the same layout as ``C``.
      * If ``B`` is transposed then we'll swap its declared dimensions
        (i.e., we'll swap its reported numbers of rows and columns) and 
        we'll tell the kernel to read it in the opposite layout as ``C``.

 3. We dispatch a kernel from ``coo_spmm_impl.hh``, or ``csc_spmm_impl.hh``,
    or ``csr_spmm_impl.hh``. The precise kernel depends on the type of ``A``, and the inferred layout for ``B``, and the declared layout for ``C``.

## Sketching dense data with sparse operators.

Sketching dense data with a sparse operator is typically handled with ``sketch_general``,
which is defined in ``skge.hh``.

If we call this function with a SparseSkOp object, ``S``, we'd immediately get routed to
either ``lskges`` or ``rskges``. Both are defined in ``skge.hh``. Here's what happens once
we're inside one of those functions.

 1. We get a COO view of ``S`` (sampling its defining data first, if that hasn't happened yet).

 2. If we've been asked for a *proper submatrix* of an already-sampled operator, we hand the
    COO view and the ``(ro_s, co_s)`` offsets straight to ``[left/right]_spmm`` and let that
    function deal with the offsets. Otherwise, we have the full operator, and we proceed to compress it.

 3. Compression and application happens in ``_[l/r]skges_compress_and_apply_coo``. These functions
    normalize-away the transposition and use a heuristic to choose the compressed format (CSR or CSC).
    This heuristic can differ from that used if we had called `[left/right]_spmm` directly on
    a COO matrix.

## SYMM-shaped kernels (spsymm)

RandBLAS exposes a SYMM-style API for sparse symmetric matrices via the
``spsymm`` family. The design covers four cases based on the structure of the
two operands (``A`` symmetric vs. the second factor ``B``):

| Tag | Operation                       | A storage           | B storage | Status                                  |
|-----|---------------------------------|---------------------|-----------|-------------------------------------------------|
| A   | dense-symm × dense              | dense, one triangle | dense     | Implemented via ``blas::symm`` in ``sksy.hh``.   |
| B   | dense-symm × sparse             | dense, one triangle | sparse    | Implemented via ``lsksys`` / ``rsksys`` wrappers in ``sksy.hh`` (validation, beta, window sampling) over the column-driven ``coo_lsksys`` kernel in ``sparse_data/coo_sksys_impl.hh`` (``coo_rsksys`` reduces to it via the transpose identity). Only the named triangle of A is read. |
| C   | sparse-symm × dense (→ dense)   | sparse, one triangle | dense     | Implemented in ``spsymm_dispatch.hh`` (side=Right normalized at entry; MKL fast path covers all three formats; column-driven per-format fallbacks). |
| D   | sparse-symm × sparse → dense    | sparse, one triangle | sparse    | Implemented in ``spsymm_dispatch.hh``: expand A's triangle to a general sparse matrix and reuse the sparse-times-sparse path (MKL builds); densify-B + Case-C composition as the non-MKL / index-width-mismatch fallback. |

### MKL availability

| Tag | MKL native?    | Notes                                                                                                           |
|-----|----------------|-----------------------------------------------------------------------------------------------------------------|
| A   | No (BLAS++)    | ``blas::symm`` directly.                                                                                        |
| B   | No             | The transpose trick puts the sparse op on the left of ``mkl_sparse_d_mm``, but the dense A has no ``matrix_descr``, so MKL can't be told A is symmetric. We hand-roll instead. See ``coo_lsksys`` / ``coo_rsksys`` in ``sparse_data/coo_sksys_impl.hh`` (with thin SparseSkOp wrappers in ``sksy.hh``). |
| C   | Yes            | ``mkl_sparse_?_mm`` with ``descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC``. side=Right is normalized to side=Left by the dispatcher before MKL is reached (layout-flip identity, valid since ``A == A^T``); CSC is consumed inside ``mkl_spsymm`` as a CSR-of-transpose view with ``uplo`` flipped. The hand kernels run only on non-MKL builds, on index-width mismatch with ``MKL_INT``, or on a runtime ``NOT_SUPPORTED``. |
| D   | Yes (via expansion) | ``mkl_sparse_sp2m`` returns ``SPARSE_STATUS_NOT_SUPPORTED`` when ``descrA.type == SPARSE_MATRIX_TYPE_SYMMETRIC`` (only ``GENERAL`` is accepted there), and ``mkl_sparse_?_spmmd`` takes no descriptor at all, so the symmetric expansion happens on the RandBLAS side: ``expand_symmetric_to_general`` (O(nnz)) followed by ``mkl_spgemm_to_dense`` with a GENERAL descriptor. |

### Case C dispatch (``spsymm_dispatch.hh``)

``RandBLAS::sparse_data::spsymm(layout, side, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy)``
dispatches as follows:

1. Normalize side=Right to side=Left: since ``A == A^T``, ``Y = B*A`` equals
   ``Y^T = A*B^T``, and reinterpreting the B and Y buffers in the opposite
   layout presents them as ``B^T`` and ``Y^T`` with the same leading
   dimensions. ``uplo`` is unchanged. Everything below assumes side=Left.
2. Validate: zero-based indices (``A.index_base == IndexBase::Zero``), A
   square of order ``m``, and leading-dimension lower bounds for B and Y
   (mirroring ``left_spmm``).
3. Handle empty products (a zero dimension, alpha == 0, or a structurally
   empty operand) by leaving beta * Y, before MKL can reject a valid empty
   sparse matrix at handle creation (the left_spmm contract).
4. If RandBLAS was built with MKL and the index width matches ``MKL_INT``,
   try ``mkl::mkl_spsymm`` with the caller's beta (MKL applies alpha and
   beta itself, as ``left_spmm`` hands beta to ``mkl_left_spmm``). It applies
   the ``SPARSE_MATRIX_TYPE_SYMMETRIC`` descriptor and calls
   ``mkl_sparse_?_mm``; CSC goes through the CSR-of-transpose view with
   ``uplo`` flipped. It returns false only on a runtime ``NOT_SUPPORTED`` (a
   parameter-validation result, so Y is untouched when it happens); control
   then falls to step 5.
5. Format-specific fallback: apply beta via ``util::lascl``, then run
   ``csr_spsymm`` / ``coo_spsymm`` (and
   ``csc_spsymm``, a three-line delegation to ``csr_spsymm`` on the
   transpose view with ``uplo`` flipped). The kernels are column-driven:
   an OpenMP-parallel outer loop over the n right-hand-side columns of B/Y
   (each column is owned by one thread, so there are no races), with a scan
   of the stored triangle inside. Each stored off-diagonal entry
   ``A(i,j) = v`` contributes twice per column (the entry and its implied
   symmetric counterpart); diagonal entries once. Entries outside the named
   triangle are silently skipped, so a caller that mistakenly stored both
   triangles still gets the correct answer.

The public-facing wrappers in the top-level ``RandBLAS::`` namespace are:

  - ``spsymm(layout, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy)``,
    convenience for side=Left.
  - ``spsymm(layout, m, n, alpha, Symmetric<SpMat> A_sym, B, ldb, beta, Y, ldy)``,
    routes via the ``Symmetric<SpMat>`` carrier so the uplo annotation
    travels with the matrix.

### Case B: column-driven COO kernel in ``coo_sksys_impl.hh``

The accumulation kernel lives in ``sparse_data/coo_sksys_impl.hh`` as
``coo_lsksys``, taking a ``COOMatrix<T, sint_t>`` plus submatrix offsets and
writing into a dense buffer. ``coo_rsksys`` is a delegation: ``B = A*S``
implies ``B^T = S^T * A`` (A symmetric), so it calls ``coo_lsksys`` with the
layout and ``uplo`` flipped, the ``Scoo.transpose()`` view, and the window
offsets swapped. Wrappers ``lsksys`` / ``rsksys`` in ``sksy.hh`` validate
the arguments (mirroring the dense-path ``lsksy3`` / ``rsksy3``), beta-scale
``B`` via ``util::lascl`` exactly once, and forward into the kernel. The
split keeps ``sksy.hh`` focused on SkOp dispatch and puts the format-specific
work next to the other COO kernels.

The kernel is column-driven: an OpenMP-parallel outer loop over the n
columns of B (each column owned by one thread, race-free), with a scan of
the window nonzeros of S inside. The contribution of nonzero
``(row_S, col_S, v)`` to column c is
``B(row_S - ro_s, c) += alpha * v * sym(A, uplo)(col_S - co_s, c)``, where
the symmetric element read resolves to the stored triangle by swapping the
index pair when it falls outside it. There is one address computation for A
(a two-way in-triangle test) instead of a per-``uplo``-per-layout grid of
strided AXPY range splits. SparseSkOp is COO internally, so submatrix
filtering is a direct ``if (row < ro_s ...) continue`` on the COO triples.

For an unmaterialized SparseSkOp, ``lsksys`` / ``rsksys`` sample only the
requested window via ``submatrix_as_coo`` (the same pattern ``lskges``
uses), so the RNG and memory cost is proportional to the window, not to the
full operator; a materialized SparseSkOp is consumed through
``coo_view_of_skop`` with the window filtered inside the kernel.

### Case D: expand-A + spgemm reuse (densify-B as the fallback)

Lives in ``spsymm_dispatch.hh`` next to the Case-C dispatcher. side=Right is
normalized to side=Left at entry via the same layout-flip identity as
Case C, with ``B^T`` obtained as the lightweight ``B.transpose()`` view.
Validation and the single beta application follow the Case-C pattern.

Why MKL cannot be handed the symmetric A directly:

  - ``mkl_sparse_sp2m`` returns ``SPARSE_STATUS_NOT_SUPPORTED`` when
    the ``matrix_descr`` on either operand is
    ``SPARSE_MATRIX_TYPE_SYMMETRIC``; only ``GENERAL`` is accepted
    there.
  - ``mkl_sparse_?_spmmd`` (which writes directly to dense ``C``)
    accepts no descriptor at all.

So the symmetric expansion happens on the RandBLAS side. Primary path
(MKL builds with both index widths matching ``MKL_INT``):
``expand_symmetric_to_general(A, uplo)`` (in ``symmetric.hh``) builds an
owning general COO from the stored triangle in ``O(nnz)`` memory, and
``mkl_spgemm_to_dense`` runs on it with a GENERAL descriptor. ``B`` stays
sparse, so the cost is proportional to the actual nonzero structure.

Fallback path (non-MKL builds, or index width mismatched with
``MKL_INT``): densify ``B`` into an ``m`` by ``n`` temporary in the
caller's ``layout`` via the format-specific ``coo_to_dense`` /
``csr_to_dense`` / ``csc_to_dense`` helper, then compose through the
Case-C overload. The ``O(m*n)`` temporary is a fallback-only cost.

## Sketching sparse data with dense operators

If we call ``sketch_sparse`` with a DenseSkOp, ``S``, and a sparse matrix, ``A``, then we'll get routed to either
``lsksp3`` or ``rsksp3``.

From there, we'll do the following.

 1. If necessary, we sample the defining data of ``S``. The way that we do this is a
    little more complicated than using ``RandBLAS::fill_dense(S)``, but it's similar
    in spirit.

 2. We get our hands on the simple buffer representation of ``S``.  From there ...
     * We call ``right_spmm`` if we're inside ``lsksp3``.
     * We call ``left_spmm`` if we're inside ``rsksp3``.
    
    Note that the ``l`` and ``r`` in the ``[l/r]sksp3`` function names
    get matched to opposite sides for ``[left/right]_spmm``! This is because all the fancy abstractions in ``S`` have been stripped away by this point in the call sequence, so the "side" that we emphasize in function names changes
    from emphasizing ``S`` to emphasizing ``A``.
