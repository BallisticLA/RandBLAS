:orphan:


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



.. ******************************************************************************
.. Implementation notes
.. ******************************************************************************

.. RNGState
.. ========

.. TODO: Implementation details. Things like the counter and key being arrays. You shouldn't need to interact with the APIs
.. of these arrays. But, to address any curiosity, one of their nice features is an ``ctr.incr(val)`` method that effectively
.. encodes addition in a way that correctly handles overflow from one entry of ``ctr`` to the next.

.. Every RNGState has an associated template parameter, RNG.
.. The default value of the RNG template parameter is :math:`\texttt{Philox4x32}`.
.. An RNG template parameter with name :math:`\texttt{GeneratorNxW}` will represent
.. the counter and key by an array of (at most) :math:`\texttt{N}` unsiged :math:`\texttt{W}`-bit integers.

.. DenseSkOp
.. ===================================


.. SparseSkOp
.. ===================================