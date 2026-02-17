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
either ``lskges`` or ``rskges``. Here's what would happen after we entered one of those functions:

 1. If necessary, we'd sample the defining data of ``S`` with ``RandBLAS::fill_sparse(S)``.

 2. We'd obtain a lightweight view of ``S`` as a COOMatrix, and we'd pass that matrix to ``left_spmm``
    (if inside ``lskges``) or ``right_spmm`` (if inside ``rskges``).


## Sketching sparse data with dense operators

If we call ``sketch_sparse`` with a DenseSkOp, ``S``, and a sparse matrix, ``A``, then  we'll get routed to either
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

