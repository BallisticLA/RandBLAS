:orphan:

.. **********************************************
.. Computing a sketch of a data matrix
.. **********************************************

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