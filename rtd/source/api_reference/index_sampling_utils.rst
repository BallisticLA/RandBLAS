
############################################################
Utilities for coordinate and index-set sampling
############################################################

    .. doxygenfunction:: RandBLAS::util::sample_indices_iid_uniform(int64_t n, int64_t k, int64_t* samples, RNGState<RNG> state)
      :project: RandBLAS

    .. doxygenfunction:: RandBLAS::util::sample_indices_iid(int64_t n, TF* cdf, int64_t k, int64_t* samples, RNGState<RNG> state)
      :project: RandBLAS

    .. doxygenfunction:: RandBLAS::util::weights_to_cdf(int64_t n, T* w, T error_if_below = -std::numeric_limits<T>::epsilon())
      :project: RandBLAS
