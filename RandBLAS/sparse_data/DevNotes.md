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

| Tag | Operation                       | A storage           | B storage | Status this PR                                  |
|-----|---------------------------------|---------------------|-----------|-------------------------------------------------|
| A   | dense-symm × dense              | dense, one triangle | dense     | Implemented via ``blas::symm`` in ``sksy.hh``    |
| B   | dense-symm × sparse             | dense, one triangle | sparse    | Stub in ``sksy.hh`` (``sketch_symmetric`` on SparseSkOp). Throws ``RandBLAS::Error``. |
| C   | sparse-symm × dense (→ dense)   | sparse, one triangle | dense     | Implemented in ``spsymm_dispatch.hh`` (MKL fast path + per-format fallbacks). |
| D   | sparse-symm × sparse → dense    | sparse, one triangle | sparse    | Stub in ``spsymm_dispatch.hh`` (overload taking two ``SparseMatrix`` args). Throws ``RandBLAS::Error``. |

### MKL availability

| Tag | MKL native?    | Notes                                                                                                           |
|-----|----------------|-----------------------------------------------------------------------------------------------------------------|
| A   | No (BLAS++)    | ``blas::symm`` directly.                                                                                        |
| B   | No             | The transpose trick puts the sparse op on the left of ``mkl_sparse_d_mm``, but the dense A has no ``matrix_descr``, so MKL can't be told A is symmetric. |
| C   | Yes (mostly)   | ``mkl_sparse_d_mm`` with ``descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC``. RandBLAS falls back to a hand kernel for side=Right (MKL has no Side parameter) and for CSC (``mkl_sparse_d_mm`` returns NOT_SUPPORTED on CSC). |
| D   | Partial        | ``mkl_sparse_sp2m`` accepts a symmetric descriptor on A but writes sparse output; densifying afterward is a workable composition fallback but not a single-call kernel. |

### Case C dispatch (``spsymm_dispatch.hh``)

``RandBLAS::sparse_data::spsymm(layout, side, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy)``
dispatches as follows:

1. Validate: A is square (``A.n_rows == A.n_cols``), and matches the side
   convention (``A.n_rows == m`` for side=Left, ``A.n_rows == n`` for side=Right).
2. If RandBLAS was built with MKL and the index width matches ``MKL_INT``,
   try ``mkl::mkl_spsymm``. It applies the ``SPARSE_MATRIX_TYPE_SYMMETRIC``
   descriptor and calls ``mkl_sparse_d_mm`` directly. Returns false for
   side=Right and CSC; control falls through to step 3 in either case.
3. Format-specific fallback: ``csr_spsymm`` / ``csc_spsymm`` / ``coo_spsymm``.
   Each iterates the named triangle once. For each stored entry ``A(i,j) = v``,
   it emits one ``blas::axpy`` for the structural location and a second one
   for the implied symmetric counterpart (when ``i != j``). Diagonal entries
   contribute once. Entries outside the named triangle are silently skipped,
   so a caller that mistakenly stored both triangles still gets the correct
   answer (the kernel just behaves as if the "extra" entries were absent).

A shared ``internal::apply_beta_scale`` helper (defined in
``csr_spsymm_impl.hh``, re-included by the other two format files) handles
the ``Y <- beta * Y`` pass on entry.

The public-facing wrappers in the top-level ``RandBLAS::`` namespace are:

  - ``spsymm(layout, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy)`` --
    convenience for side=Left.
  - ``spsymm(layout, m, n, alpha, Symmetric<SpMat> A_sym, B, ldb, beta, Y, ldy)``
    -- routes via the ``Symmetric<SpMat>`` carrier so the uplo annotation
    travels with the matrix.

### Cases B and D: why stub-only

No portable kernel for "dense-symm × sparse" or "sparse-symm × sparse → dense"
exists in MKL, Ginkgo, the SparseBLAS reference implementation, the C++
Sparse BLAS proposal (arXiv:2411.13259), or MAGMA-sparse (surveyed 2026-05).
Hand-rolling is feasible but non-trivial: case B's access pattern is "for
each nonzero of B, gather a column of A from the stored triangle", which is
awkward to vectorize; case D layers a symmetric-triangle filter on top of
an spgemm-into-dense pattern.

The stubs exist so the API surface is locked: future PRs can fill in the
bodies without breaking source compatibility for downstream callers. The
stub message points back to ``project-plans/randblas-symm-plan.md``.

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
