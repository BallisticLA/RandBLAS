FAQ and Limitations
==============================


(In)frequently asked questions about our design decisions
------------------------------------------------------------

Why the name RandBLAS?
  RandBLAS derives its name from BLAS: the basic linear algebra subprograms. Its name evokes the *purpose* of BLAS rather 
  than the acronym. Specifically, the purpose of RandBLAS is to provide high-performance and reliable functionality that 
  can be used to build sophisticated randomized algorithms for matrix computations.
  
  The RandBLAS API is also as close as possible to the BLAS API while remaining polymorphic. Some may find this
  decision dubious, since the BLAS API is notoriously cumbersome for new users. We believe that these downsides
  are made up for by the flexibilty and portability of a BLAS-like API. It is also our hope that popular high-level
  libraries that already wrap BLAS might find it straightforward to define similar wrappers for RandBLAS.

DenseDist and SparseDist are very simple immutable structs. Why bother having constructors for these classes, when you could just use initialization lists?
  DenseDist and SparseDist have common members and type-specific members.
  We wanted the common members to appear earlier initialization order, since that makes it easier to
  preserve RandBLAS' polymorphic API when called from other languages.

  Two important common members are major_axis and isometry_scale. Users rarely need to think about the former (since there are
  sensible defaults for DenseDist and SparseDist) and they never need to make decisions for the latter
  What's more, there are type-specific members that the user should be mindful of, and would be ill-suited to the trailing
  positions in an initialization list.
  
  Using constructors therefore has three benefits.

  1. We can put the type-specific mebers earlier in the argument list,
  2. We can include the default value for major_axis as a trailing constructor argument, and 
  3. We can ensure that isometry_scale is always set correctly.

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
  Makes it easier to call from languages that don't support overloads.
  Short and specific names make it possible to communicate efficiently and precisely (useful for test code and developer documentation). 

Why does sketch_symmetric not use a "side" argument, like symm in BLAS?
  There are many BLAS functions that include a "side" argument. This argument always refers to the argument with the most notable structure.
  In symm, the more structured argument is the symmetric matrix.
  In RandBLAS, the more structed argument is always the sketching operator. Given this, we saw three options to move forward.

  1. Keep "side," and have it refer to the position of the symmetric matrix. This is superficially simmilar to the underlying BLAS convention.
  2. Keep "side," and have it refer to the position of the sketching operator. This is similar to BLAS at a deeper level, but people could
     easily use it incorrectly.
  3. Dispense with "side" altogether.

  We chose the third option since that's more in line with modern APIs for BLAS-like functionality (namely, std::linalg).


(In)frequently asked questions about RandBLAS' capabilities
-----------------------------------------------------------

How do I call RandBLAS from other languages?
  First, this depends on whether your language supports overloads.

  * To get an important thing out of the way: we use both formal C++ overloads and C++ templates. The latter mechanism might as well constitute an overload from the perspective of other languages.
  * We do have canonical function names to address overloads in (1) the side of an operand in matrix-matrix products and (2) the family of sketching distribution.
  * Our templating of numerical precision should be resolved by prepending "s" for single precision or "d" for double precision on any classes and function names.

  This also depends on whether your language supports classes that can manage their own memory.

  * The full API for DenseSkOp and SparseSkOp requires letting them manage their own memory. If you use the appropriate constructors then they'll let you manage all memory.

Can functions of the form ``sketch_[xxx]`` do something other than sketching?
  Absolutely. It can do lifting, which is needed in some algorithms. It can also apply a square submatrix of a sketching operator (useful for distributed applications), in which case the output matrix isn't any smaller than the input matrix.

Can I sketch a const symmetric matrix that's only stored in an upper or lower triangle?
  Yes, but there are caveats. First, you can only use dense sketching operators. Second, you'll have to call fill_dense(layout, dist, … buff …, rngstate) and then use buff in your own SYMM function.


Unavoidable limitations
------------------------

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


Limitations of calling RandBLAS from other languages
----------------------------------------------------

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

