# Developer Notes for RandBLAS' sparse matrix functionality

RandBLAS provides basic abstractions for CSC, CSR, and COO-format sparse matrices.
The following RandBLAS functions use these abstractions either directly or indirectly:

 * ``left_spmm``, which computes a product of a sparse matrix and a dense matrix when the sparse matrix
    is the left operand. This function is GEMM-like, in that it allows offsets and transposition flags
    for either argument.
 * ``right_spmm``, which is analogous to ``left_spmm`` when the sparse matrix is the right operand.
 * ``sketch_general``, when called with a SparseSkOp object.
 * ``sketch_sparse``, when called with a DenseSkOp object.

Each of those functions is merely a dispatcher of more complicated functions. See below for details on
how the dispatching works.

## Left_spmm and right_spmm

These functions are implemented in ``RandBLAS/sparse_data/spmm_dispatch.hh``.
``right_spmm`` is implemented by falling back on ``left_spmm`` with transformed
values for ``opS, opA`` and ``layout``.

Here's what happens if ``left_spmm`` is called with a sparse matrix ``A``, a dense input matrix ``B``, and a dense output matrix ``C``.

 1. If needed, transposition of ``A`` is resolved by creating a lightweight object
    called ``At``. This object is just a tool for us to change how we intrepret the buffers that underlie ``A``.
        * If ``A`` is COO, then ``At`` will also be COO.
        * If ``A`` is CSR, then ``At`` will be CSC.
        * If ``A`` is CSC, then ``At`` will be CSR.
    We make a recursive call to ``left_spmm`` once we have our hands on ``At``, so
    the rest of ``left_spmm``'s logic only works with un-transposed ``A``.

 2. A memory layout is determined for how we'll read ``B`` in the low-level 
    sparse matrix multiplication kernels.
        * If ``B`` is un-transposed then we'll use the same layout as ``C``.
        * If ``B`` is transposed then we'll swap its declared dimensions
          (i.e., we'll swap its reported numbers of rows and columns) and 
          and we'll tell the kernel to read it in the opposite layout as ``C``.

 3. We dispatch a kernel from ``coo_spmm_impl.hh``, or ``csc_spmm_impl.hh``,
    or ``csr_spmm_impl.h``. The precise kernel depends on the type of ``A``, the declared layout for ``C``, and the inferred layout for ``B``.

## Sketching dense data with sparse operators.

Suppose we call ``sketch_general(...)`` with a SparseSkOp object, ``S``.
We'll get routed to either ``lskges(...)`` or ``rskges(...)`` in ``skges_to_spmm.hh``, 
then we'll do the following.

 0. If necessary, the defining data of ``S`` is sampled with ``RandBLAS::fill_sparse(S)``.

 1. We obtain a lightweight view of ``S`` as a COOMatrix, and we pass that matrix to ``left_spmm``
    if inside ``lskges(...)`` or ``right_spmm`` if inside ``rskges(...)``. Recall that ``right_spmm``
    falls back on an equivalent call to ``left_spmm``.

 2. ``left_spmm`` dispatches a low-level kernel based on the sparse matrix format.
    The kernels for COOMatrix objects can always be found in ``coo_spmm_impl.hh``.

 3. At time of writing, the kernel in ``coo_spmm_impl.hh`` reads the low-level data
    in ``S`` and produces data for an equivalent CSC format sparse matrix without
    using the CSCMatrix abstraction. That low-level data is passed to kernels implemented in ``csc_spmm_impl.hh``.


## Sketching sparse data with dense operators

If we call ``sketch_sparse(...)`` with a DenseSkOp object, we'll get routed to either
``lsksp3(...)`` or ``rsksp3(...)`` in ``sparse_data/sksp3_to_spmm.hh``.
Then we'll do the following.

 0. If necessary, we sample the defining data of ``S``. The way that we do this is a
    little more complicated than using ``RandBLAS::fill_dense(S)``, but it's similar
    in spirit.

 1. We get our hands on the simple buffer representation of ``S``.  From there,
    we call either ``right_spmm`` if inside ``lsksp3`` or ``left_spmm`` if inside
    ``rsksp3``.
    
        Note that the ``l`` and ``r`` in the ``[l/r]sksp3`` function names
        get matched to opposite sides for ``[left/right]_spmm``. This reflects the fact
        that all of the fancy abstractions in ``S`` have been stripped away by this point
        in the call sequence, so the "side" that we emphasize in function names changes
        from emphasizing ``S`` to emphasizing the data matrix.

    Recall that ``right_spmm`` is implemented by a simple call to ``left_spmm`` with
    transformed arguments.

 2. ``left_spmm`` dispatches a low-level kernel based on the sparse matrix format.
    We anticipate that the typical use-case for ``left_spmm`` will involve either
    a CSC or CSR format matrix. At time of writing, there are two implementations
    of the compute kernels for both of these formats. We choose between the 
    kernels based on which should exhibit better cache behavior.
