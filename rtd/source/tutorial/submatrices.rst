.. toctree::
  :maxdepth: 3

*********************************************************************************************
The meaning of "submat(・)" in RandBLAS documentation
*********************************************************************************************

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


GEMM's low-level semantics enables operating on *contiguous submatrices* of larger matrices in complicated applications.
This flexibility is a key part of its usefulness in numerical computing.
RandBLAS aims to be similarly flexible.
However, RandBLAS needs an expressive data model for its sketching operators.
It also needs to incorporate sparse data matrices.
As a result, the mechanism that GEMM uses to allow operating with submatrices is insufficient for RandBLAS.

Here we describe RandBLAS's approach to defining GEMM-like functions that allow for more abstract linear operators.
To begin, recall how the GEMM API can be described as accepting dimensions :math:`(m, n, k)`, pointers  :math:`(A, B, C)`, and executing

  .. math::
      :label: eq_realisticgemm_2
      
      \displaystyle\mat(C) = \alpha \cdot\, \underbrace{\op(\mat(A))}_{m \times k}\, \cdot \,\underbrace{\op(\mat(B))}_{k \times n} + \,\beta \cdot\underbrace{\mat(C)}_{m \times n},

where :math:`\mat(\cdot)` accepts a pointer and returns a matrix based on contextual information (inferred dimensions, stride parameters, and a layout parameter).
Recall also how :math:`\mat(\cdot)` is only an artifact of *exposition.*

It's easy to extend :eq:`eq_realisticgemm_2` to allow an abstract linear operator for :math:`\mtx{A} = \mat(A)`.
We just need an operator that selects a submatrix based on explicit row and column offset parameters.
We'll call this operator :math:`\submat(\cdot).`
Our convention is to use :math:`\text{“}(\roa, \coa)\text{”}` for the row and column offsets for :math:`\submat(\mtx{A}),`
so that 

    .. math::

        \submat(\mtx{A})_{ij} = \mtx{A}_{(i+\roa),(j+\coa)}.

With this in mind, here the natural extension of :eq:`eq_realisticgemm_2` when :math:`\mtx{A}` is an abstract
linear operator (such as a sketching operator or a sparse matrix):

  .. math::
      :label: eq_semiabstracta_gemm

      \mat(C) = \alpha \cdot\, \underbrace{\op(\submat(\mtx{A}))}_{m \times k}\, \cdot \,\underbrace{\op(\mat(B))}_{k \times n} + \,\beta \cdot \underbrace{\mat(C)}_{m \times n}.

The corresponding GEMM-like function signature is as follows.

.. code:: c++

  template <typename T, typename LinOp>
  abstract_gemm(
    blas::Layout ell, blas::Op opA, blas::Op opB, int m, int n, int k,
    T alpha, LinOp A, int ro_a, int co_a, const T* B, int ldb, T beta, T* C, int ldc
  )

Analgous changes apply just as well in two other cases: when :math:`\mtx{B}` is abstract
rather than :math:`\mtx{A}`, or when both :math:`\mtx{A}` and :math:`\mtx{B}` are abstract.

RandBLAS doesn't actually have a function called :math:`\texttt{abstract_gemm}.`
However, this pattern arises directly in RandBLAS' :math:`\texttt{sketch_general}`
functions (when one of :math:`(\mtx{A},\mtx{B})` is dense and the other is a sketching operator),
and in its :math:`\texttt{sketch_sparse}` functions 
(when one of :math:`(\mtx{A},\mtx{B})` is sparse and the other is a sketching operator).
