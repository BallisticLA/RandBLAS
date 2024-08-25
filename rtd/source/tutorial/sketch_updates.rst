

  .. |denseskop| mathmacro:: \texttt{DenseSkOp}
  .. |seedstate| mathmacro:: \texttt{seed_state}
  .. |nextstate| mathmacro:: \texttt{next_state}
  .. |mtx| mathmacro:: {}

.. _sketch_updates:

*********************************************************************************************
Updating sketches with dense sketching operators
*********************************************************************************************

RandBLAS makes it easy to *implicitly* extend an initial sketching
operator :math:`\mtx{S}_1` into an augmented operator :math:`\mtx{S} = [\mtx{S}_1; \mtx{S}_2]` or :math:`\mtx{S} = [\mtx{S}_1, \mtx{S}_2]`.
There are four scenarios that you can find yourself in where
this can be done without generating :math:`S` from scratch.
In all four scenarios, the idea is to
use the :math:`\nextstate` of :math:`\mtx{S}_1` as the
:math:`\seedstate` of :math:`\mtx{S}_2`.

There are two reasons why you'd want to
extend a sketching operator; you might be trying to improve statistical
quality by increasing sketch size, or you might be
incorporating new data into a sketch of fixed size.
The unifying perspective on the former scenarios is that they both add
*long-axis vectors* to the sketching operator.
The unifying perspective on
Scenarios 3 and 4 is that they both add *short-axis vectors* to the
sketching operator. 

:math:`\texttt{DenseDist}` objects have a :math:`\texttt{major_axis}` member, which states
whether operators sampled from that distribution are short-axis or
long-axis major. So when you specify the major axis for a sketching
operator, you're basically saying whether you want to keep open the possibility of
improving the statistical quality of a sketch or updating a sketch to
incorporate more data.


Increase sketch size by adding long-axis vectors.
=================================================

  Suppose :math:`\mtx{S}_1` is a *wide* :math:`d_1 \times m` row-wise
  :math:`\denseskop` with seed :math:`c`. It's easy to generate a 
  :math:`d_2\times m` row-wise :math:`\denseskop` :math:`\mtx{S}_2` in such a way that
  :math:`\mtx{S} = [\mtx{S}_1; \mtx{S}_2]` is the same as the :math:`(d_1 + d_2) \times m` row-wise
  :math:`\denseskop` with seed :math:`c`.

  This scenario arises if we have a fixed data matrix :math:`\mtx{A}`, an initial
  sketch :math:`\mtx{B}_1 = \mtx{S}_1 \mtx{A}`, and we decide we want a larger sketch for
  statistical reasons. The updated sketch :math:`\mtx{B} = \mtx{S} \mtx{A}` can be expressed as

    .. math::

        \mtx{B} = \begin{bmatrix} \mtx{S}_1 \mtx{A} \\ \mtx{S}_2 \mtx{A} \end{bmatrix}.

  Now suppose :math:`\mtx{S}_1`  a *tall* :math:`n \times d_1` column-wise :math:`\denseskop`
  with seed :math:`c`. We can easily generate an :math:`n\times d_2` column-wise
  :math:`\denseskop` :math:`\mtx{S}_2` so that :math:`\mtx{S} = [\mtx{S}_1, \mtx{S}_2]` is the same
  as the :math:`d \times (n_1 + n_2)` column-wise :math:`\denseskop` with seed :math:`c`.

  This arises we have a fixed data matrix :math:`\mtx{A}`, an initial sketch :math:`\mtx{B}_1 = \mtx{A} \mtx{S}_1`,
  and we decide we want a larger sketch for statistical reasons. The
  updated sketch :math:`\mtx{B} = \mtx{A}\mtx{S}` can be expressed as

    .. math::

        \mtx{B} = \begin{bmatrix} \mtx{A} \mtx{S}_1 & \mtx{A} \mtx{S}_2 \end{bmatrix}.

If :math:`(\mtx{S}_1, \mtx{S}_2, \mtx{S})` satisfy the assumptions above, then the final sketch
:math:`B` will be the same regardless of whether we computed the sketch in one
step or two steps. This is desirable for benchmarking and debugging
RandNLA algorithms where the sketch size is a tuning parameter.


Accomodate more data by adding short-axis vectors.
==================================================

  Suppose :math:`\mtx{S}_1` is a *tall* :math:`n_1 \times d` row-wise
  :math:`\denseskop` with seed :math:`c`. It's easy to generate an :math:`n_2\times d`
  row-wise :math:`\denseskop` :math:`\mtx{S}_2` in such a way that
  :math:`\mtx{S} = [\mtx{S}_1; \mtx{S}_2]` is the same as the :math:`(n_1 + n_2) \times d` row-wise
  :math:`\denseskop` with seed :math:`c`.

  This situation arises if we have an initial data matrix :math:`\mtx{A}_1`, an initial sketch 
  :math:`\mtx{B}_1 = \mtx{A}_1 \mtx{S}_1`, and then a new matrix :math:`\mtx{A}_2` arrives in such a way that we 
  want a sketch of :math:`\mtx{A} = [\mtx{A}_1, \mtx{A}_2]`. To compute :math:`\mtx{B} = \mtx{A}\mtx{S}`, we update :math:`\mtx{B}_1` 
  according to the formula

    .. math::

      \mtx{B} = \begin{bmatrix} \mtx{A}_1 & \mtx{A}_2 \end{bmatrix} \begin{bmatrix} \mtx{S}_1 \\ \mtx{S}_2 \end{bmatrix} = \mtx{B}_1 + \mtx{A}_2 \mtx{S}_2.

  Now, suppose instead :math:`\mtx{S}_1` is a *wide* :math:`d \times m_1` column-wise
  :math:`\denseskop` with seed :math:`c`. It's easy to generate a 
  :math:`d \times m_2` column-wise :math:`\denseskop` :math:`\mtx{S}_2` so that 
  :math:`\mtx{S} = [\mtx{S}_1, \mtx{S}_2]` is the same as the :math:`d \times (m_1 + m_2)` column-wise
  :math:`\denseskop` with seed :math:`c`.

  This situation arises if we have an initial data matrix :math:`\mtx{A}_1`, an
  initial sketch :math:`\mtx{B}_1 = \mtx{S}_1 \mtx{A}_1`, and then a new matrix :math:`\mtx{A}_2` arrives in
  such a way that we want a sketch of :math:`A = [\mtx{A}_1; \mtx{A}_2]`. To compute :math:`\mtx{B} = \mtx{S}\mtx{A}`, 
  we update :math:`\mtx{B}_1` according to the formula

    .. math::

      \mtx{B} = \begin{bmatrix} \mtx{S}_1 & \mtx{S}_2 \end{bmatrix} \begin{bmatrix} \mtx{A}_1 \\ \mtx{A}_2 \end{bmatrix} = \mtx{B}_1 + \mtx{S}_2 \mtx{A}_2.

If :math:`(\mtx{S}_1, \mtx{S}_2, \mtx{S})` satisfy the assumptions above, then :math:`\mtx{B}` will be the
same as though we started with all of :math:`\mtx{A}` from the very beginning. This
is useful for benchmarking and debugging RandNLA algorithms that involve
operating on data matrices that increase in size over some number of iterations.


Porque no los dos? Work with giant, implicit operators.
==========================================================

