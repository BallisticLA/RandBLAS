A GEMM-like API for abstract linear operators
=============================================

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
   .. |roa| mathmacro:: \texttt{roa}
   .. |coa| mathmacro:: \texttt{coa}
   .. |mtx| mathmacro:: \mathbf


.. .. dropdown:: A simplified version of GEMM
..     :animate: fade-in-slide-down

.. **A simplified version of GEMM**

.. dropdown:: A simplified version of GEMM
    :open:
    :animate: fade-in-slide-down

    Consider a simplified version of GEMM with conformable real linear operators :math:`(\mtx{A},\mtx{B},\mtx{C})`:

        .. math::
          :label: eq_simplifiedgemm
      
          \displaystyle\mtx{C} = \alpha \cdot\,\op(\mtx{A})\, \cdot \,\op(\mtx{B}) + \,\beta \cdot \mtx{C}.

    Here, :math:`\op(\cdot)` can return its argument either unchanged or transposed. 
    Its action on :math:`\mtx{A}` and :math:`\mtx{B}` is determined by contextual
    information in the form of flags.
    The flag for :math:`\mtx{A}` is traditionally called :math:`\text{“}\opA\text{”}` and is interpreted as

      .. math::
        \op(\mtx{A}) = \begin{cases} \mtx{A} & \text{ if } \opA \texttt{ == NoTrans}  \\ \mtx{A}^T & \text{ if } \opA \texttt{ == Trans}  \end{cases}.

    The flag for :math:`\mtx{B}` is traditionally named :math:`\text{“}\opB\text{”}` and is interpreted similarly.


.. dropdown:: An accurate description of GEMM
    :open:
    :animate: fade-in-slide-down

    The actual GEMM API accepts dimensions :math:`(m, n, k)`, pointers  :math:`(A, B, C)`, and executes

      .. math::
          :label: eq_realisticgemm
          
          \displaystyle\mat(C) = \alpha \cdot\, \underbrace{\op(\mat(A))}_{m \times k}\, \cdot \,\underbrace{\op(\mat(B))}_{k \times n} + \,\beta \cdot\underbrace{\mat(C)}_{m \times n},

    where :math:`\mat(\cdot)` accepts a pointer and returns a matrix based on the following contextual information:

      * explicit or inferred dimensions (considering :math:`\text{(}m, n, k\text{)}` and :math:`\text{(}\opA,\opB\text{)}` in :eq:`eq_realisticgemm`),
      * a stride parameter associated with the pointer, and 
      * a layout parameter that applies to all three matrices in :eq:`eq_realisticgemm`.

    We emphasize that :math:`\text{“}\mat\text{”}` only exists to help with exposition.
    No such operator appears in the GEMM API.
    For reference, here is a standard function signature for a version of GEMM that requires all three matrices
    in :eq:`eq_realisticgemm` to have a common numerical type, ``T``:

    .. code:: c++

        <template typename T>
        gemm(
          blas::Layout ell, blas::Op opA, blas::Op opB, int m, int n, int k,
          T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc
        )

    A complete explanation of how :math:`\mat` extracts submatrices from this contextual information
    is given in the dropdown below.

    .. dropdown:: Details on :math:`\mat(\cdot)`.
      :animate: fade-in-slide-down

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

.. dropdown:: Extending GEMM to abstract linear operators
  :open:
  :animate: fade-in-slide-down

  The semantics of :math:`\mat` in :eq:`eq_realisticgemm` make it possible for
  GEMM to operate on *contiguous submatrices*, and this is key to the flexibility of GEMM in numerical computing.
  RandBLAS aims to be similarly flexible.
  However, RandBLAS needs an expressive data model for its sketching operators.
  It also needs to incorporate sparse data matrices.
  As a result, the :math:`\mat` operator used to facilate working with submatrices in :eq:`eq_realisticgemm` is insufficient for RandBLAS' purposes.

  It's easy to describe a version of GEMM that supports abstract linear operators for :math:`\mtx{A}` or :math:`\mtx{B}.`
  All we need is an operator that selects a submatrix based on explicit row and column offset parameters.
  We'll call this operator :math:`\submat(\cdot).`
  Our convention is to use :math:`\text{“}(\roa, \coa)\text{”}` for the row and column offsets for :math:`\submat(\mtx{A})` in :math:`\mtx{A},`
  so that 

      .. math::

          \submat(\mtx{A})_{ij} = \mtx{A}_{(i+\roa),(j+\coa)}.

  With this in mind, here the natural extension of :eq:`eq_realisticgemm` when :math:`\mtx{A}` is an abstract
  linear operator (such as a sketching operator or a sparse matrix):

    .. math::
        :label: eq_semiabstracta_gemm

        \mat(C) = \alpha \cdot\, \underbrace{\op(\submat(\mtx{A}))}_{m \times k}\, \cdot \,\underbrace{\op(\mat(B))}_{k \times n} + \,\beta \cdot \underbrace{\mat(C)}_{m \times n}.

  The corresponding GEMM-like function signature is as follows.

  .. code:: c++

    <template typename T, typename LinOp>
    abstract_gemm(
      blas::Layout ell, blas::Op opA, blas::Op opB, int m, int n, int k,
      T alpha, LinOp A, int roa, int coa, const T* B, int ldb, T beta, T* C, int ldc
    )

  The spirit of this change to ``gemm`` applies just as well when :math:`\mtx{B}` is an abstract linear operator
  instead of :math:`\mtx{A}`, or when both :math:`\mtx{A}` and :math:`\mtx{B}` are abstract.
  Click the expandable dropdown below for more discussion.
  
  .. dropdown:: More on :math:`\submat`, :math:`\mat`, and the layout parameter.
    :animate: fade-in-slide-down

    Note that setting :math:`\roa = \coa = 0` corresponds to :math:`\submat(\mtx{A})` being a leading submatrix of :math:`\mtx{A}.`

    .... more stuff ...

    .... more stuff ...


.. .. math::

..     \mtx{A} = \begin{bmatrix} \submat(\mtx{A}) &  * \\
..                               *                & *  
..                \end{bmatrix}.

.. Alternatively, one can view the submatrix as the middle block in a :math:`3 \times 3` partition of :math:`\mtx{A}`:

..     .. math::

..       \mtx{A} = \begin{bmatrix} (\roa \times \coa)  & *                &  *  \\
..                                 *                   & \submat(\mtx{A}) &  *  \\
..                                 *                   & *                &  *  
..                   \end{bmatrix}.

.. \begin{eqnarray}
.. \mat(C) &= \alpha \cdot\, \underbrace{\op(\submat(\mtx{A}))}_{m \times k}\, \cdot \,\underbrace{\op(\mat(B))}_{k \times n} + \,\beta \cdot \underbrace{\mat(C)}_{m \times n} \\
..     \text{ and } \qquad \qquad & \text{ } \\
..   \mat(C) &= \alpha \cdot\, \underbrace{\op(\mat(A))}_{m \times k}\, \cdot \,\underbrace{\op(\submat(\mtx{B}))}_{k \times n} + \,\beta \cdot \underbrace{\mat(C)}_{m \times n}
.. \end{eqnarray}


.. These functions have the same capabilities as GEMM, in the sense that they permit operating on arbitrary contiguous submatrices.
.. However, RandBLAS uses a more abstract data model than BLAS, the way that one specifies submatrices needs to change.
.. Therefore rather than exposing a function for performing :eq:`eq_realisticgemm`, it exposes functions for performing

.. The philosophy of RandBLAS' sketching APIs
.. ==========================================

.. RandBLAS has two main functions for sketching:

..  * :math:`\texttt{sketch_general}`, which is used for dense data matrices, and 
..  * :math:`\texttt{sketch_sparse}`, which is used for sparse data matrices.

.. These functions are overloaded and templated to allow for different numerical 
.. precisions and different types of sketching operators. It's possible to apply 
.. dense or sparse sketching operators to dense matrices, and to apply dense sketching
.. operators to sparse matrices. The common thread in both
.. cases is that the final sketch is always dense.

.. From a mathematical perspective, :math:`\texttt{sketch_general}` and :math:`\texttt{sketch_sparse}`
.. have the same capabilities as GEMM.


