FAQ and Limitations
==============================



How do I do this and that?
--------------------------

How do I sketch a const symmetric matrix that's only stored in an upper or lower triangle?
  You can only do this with dense sketching operators.
  You'll have to prepare the plain buffer representation yourself with 
  :cpp:any:`RandBLAS::fill_dense_unpacked`
  and then you'll have to use that buffer in your own SYMM function.

How do I sketch a submatrix of a sparse matrix?
  You can only do this if the sparse matrix in in COO format.
  Take a look at the ``lsksp3`` and ``rsksp3`` functions in the source code (they aren't documented on this website).


Why did you ... ?
-----------------

Why the name RandBLAS?
  RandBLAS derives its name from BLAS: the basic linear algebra subprograms. Its name evokes the *purpose* of BLAS rather 
  than the acronym. Specifically, the purpose of RandBLAS is to provide high-performance and reliable functionality that 
  can be used to build sophisticated randomized algorithms for matrix computations.
  
  The RandBLAS API is also as close as possible to the BLAS API while remaining polymorphic. Some may find this
  decision dubious, since the BLAS API is notoriously cumbersome for new users. We believe that these downsides
  are made up for by the flexibilty and portability of a BLAS-like API. It is also our hope that popular high-level
  libraries that already wrap BLAS might find it straightforward to define similar wrappers for RandBLAS.

DenseDist and SparseDist are simple structs. Why bother having constructors for these classes, when you could just use initialization lists?
  Both of these types only have four user-decidable parameters.
  We tried to implement and document these structs only using four members each.
  This was doable, but very cumbersome.
  In order to write clearer documentation we introduced several additional members whose values are semantically meaningful
  but ultimately dependent on the others.
  Using constructors makes it possible for us to ensure all members are initialized consistently.

Why don't you automatically scale your sketching operators to give partial isometries in expectation?
  There are a few factors that led us to this decision. None of these factors is a huge deal, alone, but they become significant when considered together.

  1. Many randomized algorithms are actually invariant under rescaling of the sketching operators that they use internally.
  2. Sketching operators are easier to describe if we think in terms of their "natural" scales before considering their use as tools for dimension reduction.
     For example, the natural scale for DenseSkOps is to have entries that are sampled iid from a mean-zero and variance-one distribution.
     The natural scale for SparseSkOps (with major_axis==Short) is for nonzero entries to have absolute value equal to one.
     Describing these operators in their isometric scales would require that we specify dimensions, which dimension is larger than the other,
     and possibly additional tuning parameters (like vec_nnz for SparseSkOps).
  3. It's easier for us to describe how implicit concatenation of sketching operators works (see :ref:`this part <sketch_updates>` of our tutorial)
     if using the isometry scale is optional, and off by default.

Why are all dimensions 64-bit integers?
  RandNLA is interesting for large matrices. It would be too easy to have an index overflow if we allowed 32-bit indexing.
  We do allow 32-bit indexing for buffers underlying sparse matrix datastructures, but we recommend sticking with 64-bit.

I looked at the source code and found weird function names like "lskge3," "rskges," and "lsksp3." What's that about?
  There are two reasons for these kinds of names.
  First, having these names makes it easier to call RandBLAS from languages that don't support function overloading.
  Second, these short and specific names make it possible to communicate efficiently and precisely (useful for test code and developer documentation). 

Why does sketch_symmetric not use a "side" argument, like symm in BLAS?
  There are many BLAS functions that include a "side" argument. This argument always refers to the argument with the most notable structure.
  In symm, the more structured argument is the symmetric matrix.
  In RandBLAS, the more structed argument is always the sketching operator. Given this, we saw three options to move forward.

  1. Keep "side," and have it refer to the position of the symmetric matrix. This is superficially simmilar to the underlying BLAS convention.
  2. Keep "side," and have it refer to the position of the sketching operator. This is similar to BLAS at a deeper level, but people could
     easily use it incorrectly.
  3. Dispense with "side" altogether.

  We chose the third option since that's more in line with modern APIs for BLAS-like functionality (namely, std::linalg).


Limitations
-----------

No complex domain support:
  BLAS' support for this is incomplete. You can't mix real and complex, you can't conjugate without transposing, etc… 
  We plan on revisiting the question of complex data in RandBLAS a few years from now.

No support for sparse-times-sparse (aka SpGEMM):
  This will probably "always" be the case, since we think it's valuable to keep RandBLAS' scope limited.

No support for subrampled randomized trig transforms (SRFT, SRHT, SRCT, etc...):
  We'd happily accept a contribution of a randomized Hadamard transform (without subsampling)
  that implicitly zero-pads inputs when needed. Given such a function we could figure out 
  how we'd like to build sketching operators on top of it.

No support for DenseSkOps with Rademachers:
  We'd probably need support for mixed-precision arithmetic to fully realize the advantage of
  Rademacher over uniform [-1,1]. It's not clear to me how we'd go about doing that. There 
  *is* the possibility of generating Rademachers far faster than uniform [-1, 1]. The implementation
  of this method might be a little complicated. 

No support for negative values of "incx" or "incy" in sketch_vector.
  The BLAS function GEMV supports negative strides between input and output vector elements.
  It would be easy to extend sketch_vector to support this if we had a proper
  SPMV implementation that supported negative increments. If someone wants to volunteer 
  to extent our SPMV kernels to support that, then we'd happily accept such a contribution.
  (It shouldn't be hard! We just haven't gotten around to this.)

Symmetric matrices have to be stored as general matrices.
  This stems partly from a desire to have sketch_symmetric work equally well with DenseSkOp and SparseSkOp.
  Another reason is that BLAS' SYMM function doesn't allow transposes, which is a key tool we use
  in sketch_general to resolve layout discrepencies between the various arguments.


Language interoperability
-------------------------

C++ idioms and features we do use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Things that affect our API:
 * Templates. We template for floating point precision just about everywhere.
   We also template for stateful random number generators (see :cpp:any:`RandBLAS::RNGState`)
   and arrays of 32-bit versus 64-bit signed integers.
 * Standard constructors. We use these for any nontrivial struct type in RandBLAS. They're important
   because many of our datatypes have const members that need to be initialized as functions (albeit
   simple funcitons) of other members.
 * Move constructors. We use these to return nontrivial datastructures from a few undocumented functions.
   We mostly added them because we figured users would really want them.
 * C++20 Concepts. These make our assumptions around template parameters more explicit.
   In the cases of :ref:`SketchingDistribution <concept_rand_b_l_a_s_1_1_sketching_distribution>` and
   :ref:`SketchingOperator <concept_rand_b_l_a_s_1_1_sketching_operator>` this is also a way
   for us to declare a common interface for future functionality.

Things that are purely internal:
 * C++17 ``if constexpr`` branching.
 * Structured bindings. 


C++ idioms and features we don't use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * The span or mdspan datastructures.
 * Inheritance.
 * Private or protected members of structs.
 * Shared pointers.
 * Instance methods for structs (with the exceptions of constructors and destructors).

Naming conventions to resolve function overloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


We routinely use function overloading, and that reduces portability across languages.
See below for details on where we stand and where we plan to go to resolve this shortcoming.

We have a consistent naming convention for functions that involve sketching operators
 * [L/R] are prefixes used when we need to consider left and right-multiplication.
 * The characters "sk" appearing at the start of a name or after [L/R] indicates that a function involves taking a product with a sketching operator.
 * Two characters are used to indicate the structure of the data in the sketching operatation.
   The options for the characters are {ge, sy, ve, sp}, which stand for general, *explicitly* symmetric, vector, and sparse (respectively).
 * A single-character [X] suffix is used to indicate the structure of the sketching operator. The characters are "3" (for dense sketching
   operators, which would traditionally be applied with BLAS 3 function) and "s" (for sparse sketching operators).

Functions that implement the overload-free conventions
 * [L/R]skge[X] for sketching a general matrix from the left (L) or right (R) with a matrix whose structure is indicated by [X].
   C++ code should prefer overloaded sketch_general
 * [L/R]sksp3 for sketching a sparse matrix from the left (L) (L) or right (R) with a DenseSkOp.
   C++ code should prefer overloaded sketch_sparse, unless operating on a submatrix of a COO-format sparse data matrix is needed.

Functions that are missing implementations of this convention
 * [L/R]skve[X] for sketching vectors. This functionality is availabile in C++ with sketch_vector
 * [L/R]sksy[X] for sketching a matrix with *explicit symmetry*. This functionality is availabile in C++ with sketch_symmetric.

Some discussion

  Our templating for numerical precision should be resolved by prepending "d" for double precision or "s" for single precision

  RandBLAS requires a consistent naming convention across an API that supports multiple structured operands (e.g., sketching sparse data),
  while conventions in the BLAS API only need to work when one operand is structured.
  This is why our consistent naming convention might appear "less BLAS-like" than it could be.

  All of these overload-free function names have explicit row and column offset parameters to handle submatrices of linear operators.
  However, the overloaded versions of these functions have *additional* overloads based on setting the offset parameters to zero.

We have no plans for consistent naming of overload-free sparse BLAS functions. The most we do in this regard is offer functions
called [left/right]_spmm for SpMM where the sparse matrix operand appears on the left or on the right.

