   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |lda| mathmacro:: \texttt{lda}
   .. |mtxA| mathmacro:: \mathbf{A}
   .. |ttt| mathmacro:: \texttt



############################################################
Utilities
############################################################

Random sampling from index sets
===============================

.. doxygenfunction:: RandBLAS::weights_to_cdf(int64_t n, T* w, T error_if_below = -std::numeric_limits<T>::epsilon())
  :project: RandBLAS

.. doxygenfunction:: RandBLAS::sample_indices_iid(int64_t n, const T* cdf, int64_t k, sint_t* samples, const state_t &state)
  :project: RandBLAS

.. doxygenfunction:: RandBLAS::sample_indices_iid_uniform(int64_t n, int64_t k, sint_t* samples, const state_t &state)
  :project: RandBLAS

.. doxygenfunction:: RandBLAS::repeated_fisher_yates(int64_t k, int64_t n, int64_t r, sint_t *samples, const state_t &state)
  :project: RandBLAS 

Debugging
=========

.. doxygenfunction:: RandBLAS::print_colmaj
   :project: RandBLAS

.. doxygenfunction:: RandBLAS::typeinfo_as_string()
   :project: RandBLAS

