Sketching dense data matrices in RandBLAS
=========================================

RandBLAS currently supports sketching dense data matrices from the left with
dense or sparse sketching operators.
This is done with the functions LSKGE3 and LSKGES, respectively.
RandBLAS' API is designed so that the interfaces for performing these tasks
are virtually identical, even though the implementations of these functions
differ substantially.

The generic format for a sketching with numeric datatype ``T`` and a sketching
operator of type ``SKOP`` is as follows::

//    LSKGEX(blas::Layout layout, blas::Op transS, blas::Op transA,
//           d, n, m, 
//           T alpha, SKOP &S, i_os, j_os, const T A, lda,
//           T beta, T B, ldb
//    )


This performs the operation::

//     mat(B) = alpha op(submat(S)) \times op(mat(A)) + beta mat(B).


where ``alpha`` and ``beta`` are real scalars and ``op(X)`` either returns
the matrix ``X`` or its transpose.

This operation involves three matrices: ``submat(S)``, ``mat(A)``, and ``mat(B)``.
The definitions of these matrices depend on other arguments in a way
that is similar to GEMM in BLAS.
We explanation these definitions below

Matrix shapes
-------------
The dimensions of ``submat(S)``, ``mat(A)``, and ``mat(B)`` are defined by

  *  ``op(submat(S))``   is d-by-m,
  *  ``op(mat(A))``      is m-by-n, and
  *  ``mat(B)``          is d-by-n.

Note that only ``mat(B)`` has its shape given explicitly. The shape of
``submat(S)`` can be ``(d, m)`` or ``(m, d)``, depending on the value of ``transS``.
Similarly, the shape of ``mat(A)`` can be ``(m, n)`` or ``(n, m)``, depending on
the value of ``transA``.

Matrix contents
---------------

The contents of ``mat(A)`` and ``mat(B)`` are determined by

  1. their shapes,
  2. the 1D arrays pointed to by ``A`` and ``B``,
  3. the matrix storage specification "layout", and
  4. the stride parameters ``(lda, ldb)``.

We interpret this information in the same way that BLAS would.
For example, here is how we define ``(A, mat(A))``:

  * If ``layout == ColMajor``, then ``mat(A)[i, j] = A[i + j*lda]``.
    In this case, ``lda`` must be >= the length of a column in ``mat(A)``.
  
  * If ``layout == RowMajor``, then ``mat(A)[i, j] = A[i*lda + j]``.
    In this case, ``lda`` must be >= the length of a row in ``mat(A)``.

The contents of ``submat(S)`` are determined by

  1. its shape,
  2. the sketching operator ``S``,
  3. a row offset parameter ``i_os``,
  4. a column offset parameter ``j_os``

If the number rows and columns in ``submat(S)`` are denoted by ``(r, c)``,
then ``submat(S)`` is the r-by-c submatrix of ``S`` whose upper-left corner
appears at index ``(i_os, j_os)`` of ``S``.

Note: the ability to select submatrices of the form described above
is *identical* to the ability to select submatrices in BLAS.