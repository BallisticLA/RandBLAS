
  .. |op| mathmacro:: \operatorname{op}
  .. |mat| mathmacro:: \operatorname{mat}
  .. |submat| mathmacro:: \operatorname{submat}
  .. |lda| mathmacro:: \texttt{lda}
  .. |ldb| mathmacro:: \texttt{ldb}
  .. |ldc| mathmacro:: \texttt{ldc}
  .. |ldx| mathmacro:: \texttt{ldx}
  .. |opA| mathmacro:: \texttt{opA}
  .. |opB| mathmacro:: \texttt{opB}
  .. |opS| mathmacro:: \texttt{opS}
  .. |roa| mathmacro:: \texttt{ro_a}
  .. |coa| mathmacro:: \texttt{co_a}
  .. |mtx| mathmacro:: \mathbf


.. _gemm_tutorial:

*******************************
Background on GEMM
*******************************

.. toctree::
  :maxdepth: 3



A simplified viewpoint
======================
Consider a simplified version of GEMM with conformable real linear operators :math:`(\mtx{A},\mtx{B},\mtx{C})`:

    .. math::
  
      \displaystyle\mtx{C} = \alpha \cdot\,\op(\mtx{A})\, \cdot \,\op(\mtx{B}) + \,\beta \cdot \mtx{C}.

Here, :math:`\op(\cdot)` can return its argument either unchanged or transposed. 
Its action on :math:`\mtx{A}` and :math:`\mtx{B}` is determined by contextual
information in the form of flags.
The flag for :math:`\mtx{A}` is traditionally called :math:`\text{“}\opA\text{”}` and is interpreted as

  .. math::
    \op(\mtx{A}) = \begin{cases} \mtx{A} & \text{ if } \opA \texttt{ == NoTrans}  \\ \mtx{A}^T & \text{ if } \opA \texttt{ == Trans}  \end{cases}.

The flag for :math:`\mtx{B}` is traditionally named :math:`\text{“}\opB\text{”}` and is interpreted similarly.

An accurate description
===========================
The GEMM API accepts dimensions :math:`(m, n, k)`, pointers  :math:`(A, B, C)`, and executes

  .. math::
      :label: eq_realisticgemm
      
      \displaystyle\mat(C) = \alpha \cdot\, \underbrace{\op(\mat(A))}_{m \times k}\, \cdot \,\underbrace{\op(\mat(B))}_{k \times n} + \,\beta \cdot\underbrace{\mat(C)}_{m \times n},

where :math:`\mat(\cdot)` accepts a pointer and returns a matrix based on the following contextual information:

  * explicit or inferred dimensions (considering :math:`\text{(}m, n, k\text{)}` and :math:`\text{(}\opA,\opB\text{)}` in :eq:`eq_realisticgemm`),
  * a stride parameter associated with the pointer, and 
  * a layout parameter that applies to all three matrices in :eq:`eq_realisticgemm`.

We use the :math:`\text{“}\mat\text{”}` operator only to help with exposition.
No such operator appears in the GEMM API.
For reference, here is a standard function signature for a version of GEMM that requires all three matrices
in :eq:`eq_realisticgemm` to have a common numerical type, ``T``.

.. code:: c++

    template <typename T>
    gemm(
      blas::Layout ell, blas::Op opA, blas::Op opB, int m, int n, int k,
      T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc
    )

A complete explanation of how :math:`\mat` extracts submatrices from this contextual information
is given below.


Details on :math:`\mat(\cdot)`
------------------------------

The semantics of :math:`\mat` can be understood by focusing on :math:`\mtx{A} = \mat(A)`.
First, there is the matter of the dimensions.
These are inferred from :math:`(m, k)` and from :math:`\opA` in the way indicated by :eq:`eq_realisticgemm`.

* If :math:`\opA \texttt{ == NoTrans}`, then :math:`\mtx{A}` is :math:`m \times k`.
* If :math:`\opA \texttt{ == Trans }`, then :math:`\mtx{A}` is :math:`k \times m`.

Moving forward let us say that :math:`\mtx{A}` is :math:`r \times c`.
The actual contents of :math:`\mtx{A}` are determined by the pointer, :math:`A\text{,}`
an explicitly declared stride parameter, :math:`\lda\text{,}`
and a layout parameter, :math:`\texttt{ell}\text{,}` according to the rule 

  .. math::
      \mtx{A}_{i,j} = \begin{cases}  A[\,i + j \cdot \lda\,] & \text{ if } \texttt{ell == ColMajor} \\ A[\,i \cdot \lda + j\,] & \text{ if } \texttt{ell == RowMajor} \end{cases}

where we zero-index :math:`\mtx{A}` for consistency with indexing into buffers in C/C++.

Only the leading :math:`r \times c` submatrix of :math:`\mat(A)` will be accessed in computing :eq:`eq_realisticgemm`.
Note that in order for this submatrix to be well-defined it's necessary that

  .. math::
    \lda \leq \begin{cases} r & \text{ if } \texttt{ell == ColMajor} \\  c & \text{ if } \texttt{ell == RowMajor} \end{cases}.

Most performance libraries check that this is the case on entry to GEMM and will raise an error if this condition
isn't satisfied.
