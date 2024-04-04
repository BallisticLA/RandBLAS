:orphan:


******************************************************************************
Implementation notes
******************************************************************************

RNGState
========

TODO: Implementation details. Things like the counter and key being arrays. You shouldn't need to interact with the APIs
of these arrays. But, to address any curiosity, one of their nice features is an ``ctr.incr(val)`` method that effectively
encodes addition in a way that correctly handles overflow from one entry of ``ctr`` to the next.

Every RNGState has an associated template parameter, RNG.
The default value of the RNG template parameter is :math:`\texttt{Philox4x32}`.
An RNG template parameter with name :math:`\texttt{GeneratorNxW}` will represent
the counter and key by an array of (at most) :math:`\texttt{N}` unsiged :math:`\texttt{W}`-bit integers.

DenseSkOp
===================================


SparseSkOp
===================================