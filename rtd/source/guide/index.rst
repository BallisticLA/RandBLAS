Sketching dense data matrices in RandBLAS
=========================================

.. note::

    TODO: reformat all the ``\math{}`` commands to use rst ``:math:``.

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

This operation involves three matrices: \math{\submat(S)}, \math{\mat(A)}, and \math{\mat(B)}.
The definitions of these matrices depend on other arguments in a way
that is similar to GEMM in BLAS.

**Matrix shapes**

The dimensions of \math{\submat(S)}, \math{\mat(A)}, and \math{\mat(B)} are defined by

 * \math{\op(\submat(S))} is  \math{d \times n},
 * \math{\op(\mat(A))}    is  \math{m \times n}, and
 * \math{\mat(B)}         is  \math{d \times n}.

Note that only \math{\mat(B)} has its shape given explicitly. The shape of
\math{\submat(S)} can be \math{(d, m)} or \math{(m, d)} depending on the value of transS.
Similarly, the shape of \math{\mat(A)} can be \math{(m, n)} or \math{(n, m)} depending on
the value of transA.

**Dense matrix contents**

The contents of \math{\mat(A)} and \math{\mat(B)} are determined by

 1. the 1D arrays pointed to by \math{A} and \math{B}
 2. the matrix storage specification "layout", and
 3. the stride parameters \math{(\lda, \ldb)}.

We interpret this information in the same way that BLAS would.
For example, here is how we define \math{(A, \mat(A))} :

 * If layout == ColMajor, then
   @verbatim embed:rst:leading-slashes
    .. math::
         \mat(A)[i, j] = A[i + j \cdot \lda].
   @endverbatim
   In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
    
 * If layout == RowMajor, then
   @verbatim embed:rst:leading-slashes
    .. math::
         \mat(A)[i, j] = A[i \cdot \lda + j].
   @endverbatim
   In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.

**Sparse matrix contents**

The contents of \math{\submat(S)} are determined by

 1. the SparseSkOp object \math{S},
 2. a row offset parameter "i_os",
 3. a column offset parameter "j_os"

If the number rows and columns in \math{\submat(S)} are denoted by \math{(r, c)},
then \math{\submat(S)} is the r-by-c submatrix of \math{S} whose upper-left corner
appears at index (i_os, j_os) of \math{S}.

Note: the ability to select submatrices of the form described above
is *identical* to the ability to select submatrices in BLAS.


