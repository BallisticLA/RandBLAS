.. toctree::
  :maxdepth: 3

*******************************
Working with submatrices
*******************************

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
As a result, the :math:`\mat` operator used to facilate working with submatrices in :eq:`eq_realisticgemm` is insufficient for RandBLAS' purposes.

Luckily, it's easy to describe a version of GEMM that supports abstract linear operators for :math:`\mtx{A}` or :math:`\mtx{B}.`
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

Analgous changes apply just as well in two other cases: when :math:`\mtx{B}` is abstract
rather than :math:`\mtx{A}`, or when both :math:`\mtx{A}` and :math:`\mtx{B}` are abstract.
Click the dropdown below for more discussion.

.. dropdown:: More on :math:`\submat`, :math:`\mat`, and the layout parameter.
  :animate: fade-in-slide-down

  Note that setting :math:`\roa = \coa = 0` corresponds to :math:`\submat(\mtx{A})` being a leading submatrix of :math:`\mtx{A}.`

  .... more stuff ...

  .... more stuff ...
