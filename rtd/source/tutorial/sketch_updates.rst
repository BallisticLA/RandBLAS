

  .. |seedstate| mathmacro:: \texttt{seed_state}
  .. |nextstate| mathmacro:: \texttt{next_state}
  .. |majoraxis| mathmacro:: \texttt{major_axis}
  .. |ttt| mathmacro:: \texttt
  .. |D| mathmacro:: \mathcal{D}
  .. |mS| mathmacro:: {S}
  .. |mA| mathmacro:: {A}
  .. |mB| mathmacro:: {B}
  .. |mX| mathmacro:: {X}
  .. |mY| mathmacro:: {Y}
  .. |R| mathmacro:: \mathbb{R}
  .. |rank| mathmacro:: \operatorname{rank}

.. _sketch_updates:


*********************************************************************************************
Updating and downdating sketches
*********************************************************************************************

This page presents four ways of updating a sketch.
We use MATLAB notation for in-line concatenation of matrices.


Increasing sketch size 
-----------------------

The scenarios here use sketching operators :math:`S_1` and :math:`S_2` 
that are sampled independently from distributions :math:`\D_1` and :math:`\D_2.`
We denote the isometry scales of :math:`\D_1` and :math:`\D_2` by :math:`\alpha_1` and :math:`\alpha_2,` respectively.

Increasing the size of a sketch is glorified concatenation.
The only subtlety is how to perform the update in a way that preserves isometric scaling
(which can be useful in contexts like norm estimation).

Our main purpose in explaining these updates is to highlight the effects of 
setting a distribution's :math:`\majoraxis` to Long.
If you're working with DenseDists and DenseSkOps (where there's no statistical difference
between short-axis-major or long-axis-major) then this choice of major axis (which is
the default) can provide an additional measure of reproducibility in experiments that require
tuning sketch size.


Sketching from the left
~~~~~~~~~~~~~~~~~~~~~~~
Here the :math:`\D_i` are distributions over wide matrices, and we set :math:`\mB_i = \mS_i \mA.`
To increase sketch size is to combine these individual sketches via concatenation:

.. math::
  \mB = \begin{bmatrix} \mB_1 \\ \mB_2 \end{bmatrix} = \begin{bmatrix} \mS_1 \mA \\ \mS_2 \mA \end{bmatrix}.

It only makes sense to do this if :math:`\mB` ends up having fewer rows than :math:`\mA.`
Put another way, this type of update only makes sense if the block operator
defined by :math:`\mS = [\mS_1;~\mS_2]` is also wide.

It is important to be aware of the basic statistical properties of this block operator,
so we'll give its distribution the name :math:`\D.`
The isometry scale of :math:`\D` is :math:`\alpha = (\alpha_1^{-2} + \alpha_2^{-2})^{-1/2}.`
If :math:`B_1` was computed with isometric scaling (that is, if :math:`\mB_1 = \alpha_1 \mS_1 \mA`),
then isometrically-scaled
updated sketch would be :math:`B = \alpha [ \mB_1/\alpha_1;~ \mB_2].`

RandBLAS can explicitly represent :math:`\D` under certain conditions, which we express with a code snippet.

.. code:: c++

  // SkDist is either DenseDist or SparseDist.
  // (d1, d2, and m) are positive integers where d1 + d2 < m.
  // arg3 is any variable allowed in the third argument of the SkDist constructor.
  SkDist D1(     d1, m, arg3, Axis::Long);
  SkDist D2(     d2, m, arg3, Axis::Long);
  SkDist  D(d1 + d2, m, arg3, Axis::Long);


Furthermore, if :math:`\mS_1.\nextstate` is the seed state for :math:`\mS_2`, then
the resulting block operator :math:`\mS = [\mS_1;~\mS_2]` equals the operator obtained by sampling from
:math:`\D` with :math:`\mS_1.\seedstate.`

This presents another option for how sketch updates might be performed.
Rather than working with two sketching operators explicitly,
one can work with a single operator :math:`\mS` sampled from the larger
distribution :math:`\D,` and compute :math:`\mB_1` and :math:`\mB_2` by working with
appropriate submatrices of :math:`\mS.`

Sketching from the right
~~~~~~~~~~~~~~~~~~~~~~~~
Here the :math:`\D_i` are distributions over tall matrices, and :math:`\mB_i = A \mS_i.`
The combined sketch is

.. math::

    \mB = \begin{bmatrix} \mB_1 & \mB_2 \end{bmatrix} = \begin{bmatrix} \mA \mS_1 & \mA \mS_2 \end{bmatrix},

and it can be obtained by right-multiplying :math:`\mA` with the block operator 
:math:`\mS = [\mS_1,~\mS_2].`

The isometry scale of :math:`\mS`'s distribution is the same as before: :math:`\alpha = (\alpha_1^{-2} + \alpha_2^{-2})^{-1/2},`
and RandBLAS can explicitly represent this distribution under the following conditions
(:math:`\ttt{SkDist}` and :math:`\ttt{x}` are as before).

.. code:: c++

  // (d1, d2, n) are positive integers where d1 + d2 < n.
  SkDist D1(n,      d1, arg3, Axis::Long);
  SkDist D2(n,      d2, arg3, Axis::Long);
  SkDist  D(n, d1 + d2, arg3, Axis::Long);

If :math:`\mS_1` is sampled from :math:`\D_1` with seed state :math:`r` and 
:math:`\mS_2` is sampled from :math:`\D_2` with seed state equal to :math:`\mS_1.\nextstate,`
then the block operator :math:`\mS` is the same as the matrix sampled from 
:math:`\D` with seed state :math:`r.`
As with sketching from the left, this shows there are situations where
it can suffice to define a single operator and sketch with appropriate submatrices
of that operator.


Rank-:math:`k` updates
----------------------

A *rank-k update* is a multiply-accumulate operation involving matrices. 
It involves a pair of matrices :math:`\mX` and :math:`\mY` that have k columns and k rows, respectively.
It also involves a real scalar :math:`\alpha` and a matrix :math:`\mB` of the same shape as :math:`\mX \mY.`
The operation itself is 

.. math::
    \mB \leftarrow \mB  + \alpha \mX \mY.

Here we describe some rank-k updates that arise in sketching algorithms.

This framework can be used to describe incorporating new data into a sketch,
or removing the contributions of old data from a sketch.
We've focused our documentation efforts on the cases that add data.
More specifically, we focus on when we're performing a rank-k update to add new
data into an existing sketch, but k was not known when the original sketch was
formed. 
This case has more complications than if k was known in advance, but it can still be handled
with RandBLAS when using distributions with *if* :math:`\majoraxis = \ttt{Short}.`

.. note::
  Future *updates* (pun intended) to these web docs will explain how the major-axis requirement
  can be dropped if an upper bound on k is known in advance. That really just amounts to explaining
  in detail how you operate with submatrices in RandBLAS. Incidentally, operating with submatrices
  is really all you need to perform rank-k updates that "remove" data.


Adding data: left-sketching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem statement
  We start with an :math:`m \times n` data matrix :math:`\mA_1.`
  When presented with this matrix, we sample a wide operator :math:`\mS_1` from a distribution :math:`\D_1`
  (defined as follows)

  .. code:: c++
  
      // Assumptions
      //  * SkDist is DenseDist or SparseDist
      //  * arg3 is any variable that makes sense for the SkDist constructor.
      //  * This example requires d < m.
      SkDist D1(d, m, arg3, Axis::Short);

  and we compute a sketch :math:`\mB = \mS_1 \mA_1.`

  Sometime later, a :math:`k \times n` matrix :math:`\mA_2` arrives.
  We want to independently sample a :math:`d \times k` operator :math:`\mS_2` from *some* :math:`\D_2` and perform a rank-k update :math:`\mB \leftarrow \mB + \mS_2 \mA_2.`
  In essence, we want to redefine

  .. math::

      \mB = \begin{bmatrix} \mS_1 & \mS_2 \end{bmatrix} \begin{bmatrix} \mA_1 \\ \mA_2 \end{bmatrix}

  without having to revisit :math:`\mA_1.`

Conceptual solution
  Since :math:`\majoraxis` is Short and :math:`d < m,` the columns of :math:`d \times m` matrices sampled from :math:`\D_1`
  will be sampled independently from a shared distribution on :math:`\mathbb{R}^d.`
  This suggests we could define :math:`\D_2` as a distribution over :math:`d \times k` matrices whose columns follow the same
  distribution used in :math:`\D_1.`

Implementation
  If :math:`d < k,` then short-axis vectors of a :math:`d \times k` matrix still refer to columns.
  This makes it possible to express :math:`\D_2` explicitly:
  
  .. code:: c++
    
    SkDist D2(d, k, arg3, Axis::Short);
  
  If :math:`d \geq k,` then we have to define a distribution :math:`\D` over :math:`d \times (m + k)` matrices  

  .. code:: c++

      SkDist D(d, m + k, arg3, Axis::Short);

  and think of :math:`\D_2` as the distribution obtained by selecting the trailing :math:`k` columns of a sample from :math:`\D.`

  This second approach may look wasteful, but that's not really the case.
  If a DenseSkOp is used in one of RandBLAS' functions for sketching with a specified submatrix,
  only the submatrix that's necessary for the operation will be generated.
  The following code snippet provides more insight on the situation.

  .. code:: c++

      SkDist D1( d,     m, arg3, Axis::Short );
      SkDist D(  d, m + k, arg3, Axis::Short ); 
      // Since d < m and we're short-axis major, the columns of matrices sampled from
      // D1 or D1 will be sampled i.i.d. from some distribution on R^d.

      auto S1 = D1.sample( seed_state ); // seed_state is some RNGState.
      auto S  =  D.sample( seed_state );
      // With these definitions, S1 is *always* equal to the first m columns of S.
      // We recover S2 by working implicitly with the trailing k columns of S.


Adding data: right-sketching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem statement
  We start with an :math:`m \times n` data matrix :math:`\mA_1.`
  When presented with this matrix, we sample a tall operator :math:`\mS_1` from a distribution :math:`\D_1` of the form

  .. code:: c++
  
      // Assumptions
      //  * SkDist is DenseDist or SparseDist
      //  * arg3 is any variable that makes sense for the SkDist constructor.
      //  * This example requires n > d.
      SkDist D1(n, d, arg3, Axis::Short);

  and we compute a sketch :math:`\mB = \mA_1 \mS_1.`

  Sometime later, an :math:`m \times k` matrix :math:`\mA_2` arrives.
  We want to independently sample an :math:`k \times d` operator :math:`\mS_2` from *some* :math:`\D_2`
  and perform a rank-k update :math:`\mB \leftarrow \mB + \mA_2 \mS_2.`
  Essentially, we want to redefine

  .. math::

      \mB = \begin{bmatrix} \mA_1 & \mA_2 \end{bmatrix}\begin{bmatrix} \mS_1 \\ \mS_2 \end{bmatrix} 

  without having to revisit :math:`\mA_1.`

Conceptual solution
  The idea is the same as with left-sketching. The difference is that since we're sketching from the
  right with a tall :math:`n \times d` operator, the short-axis vectors are rows instead of columns.
  This means the rows of :math:`\mS_1` are sampled independently from some distribution on :math:`\R^d,`
  and we can define :math:`\mS_2` by sampling its rows from that same distribution.

Implementation
  If :math:`k > d,` then we can represent :math:`\D_2` explicitly, constructing it as follows.
  
  .. code:: c++
    
    SkDist D2(k, d, arg3, Axis::Short);
  
  If :math:`k \leq d,` then we have to define a distribution :math:`\D` over :math:`(n + k) \times d` matrices  

  .. code:: c++

      SkDist D(n + k, d, arg3, Axis::Short);

  and think of :math:`\D_2` as the distribution obtained by selecting the bottom :math:`k` rows of a sample from :math:`\D.`

  As with the left-sketching case, we provide a code snippet to summarize the situation.
  
  .. code:: c++

      SkDist D1(     n, d, arg3, Axis::Short );
      SkDist D(  n + k, d, arg3, Axis::Short ); 
      // Since n > d and we're short-axis major, the rows of matrices sampled from
      // D1 or D1 will be sampled i.i.d. from some distribution on R^d.

      auto S1 = D1.sample( seed_state ); // seed_state is some RNGState.
      auto S  =  D.sample( seed_state );
      // With these definitions, S1 is *always* equal to the first m rows of S.
      // We recover S2 by working implicitly with the last k rows of S.

