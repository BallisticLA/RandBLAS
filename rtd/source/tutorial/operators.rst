.. :sd_hide_title:

.. toctree::
  :maxdepth: 3

***********************************************
Sketching distributions and sketching operators
***********************************************

The most important thing about sketching operators in RandBLAS
is simply their mathematical meaning. Once you understand that
meaning it's extremely easy to construct a sketching
operator and use it to compute a sketch.
Therefore this page starts with the mathematical meanings of the
dense and sparse sketching operators in RandBLAS.



Dense sketching operators
=========================



Sparse sketching operators
==========================


Data structures and memory management
==================================================

.. note::
  TODO: have relevant parts of the API docs link back here for a longer discussion
  of the MajorAxis concept.

.. note::
    Sketching operators in RandBLAS have a "MajorAxis" member.
    The semantics of this member can be complicated.
    We only expect advanced users to benefit from chosing this member
    differently from the defaults we set.
    We discuss the deeper meaning of and motivation for this member
    later on this page.
