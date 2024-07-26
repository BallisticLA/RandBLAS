# Developer notes for RandBLAS' testing infrastructure


This document doesn't don't defend previous design decisions.
It just explains how things work right now.
That's easier for me (Riley) to write, and it's more useful to others.
(Plus, it helps make the pros and cons of the current approach self-evident.)

None of our testing infrastructure is considered part of the public API.

## Contents

### matmul_wrappers

  * sketch_vector. It reduces to the same sketch_general no matter the type of the sketching operator.
  * sketch_sparse. It reduces to left_spmm/right_spmm no matter the type of the data matrix. (The
    sketching operator type is naturally fixed to DenseSkOp.)
  * sketch_symmetric. It reduces to the same sketch_general no matter the type of the sketching operator.

### matmul_cores

  * lskges, rskges, lskge3, rskge3. The rskgex functions could reduce to lskgex by transposing the
    product and flipping the layout. Strictly speaking, the rskgex functions don't do that, but they
    easily could. In any case, we currently have similar tests for rskgex and lskgex.
  * left_spmm and right_spmm. The right_spmm implementation falls back on left_spmm. Despite this,
    right_spmm has its own set of tests.

I suspect that the tests for rskgex and right_spmm hit code paths that are currently untested,
but I haven't actually verified this. 

### test_basic_rng

  * test_r123_kat.cc has deterministic tests for Random123. The tests comapre generated values
    to reference values computed ahead of time. The tests are __extremely__ messy, since they're
    adapted from tests in the official Random123 repository, and Random123 needs to handle a far wider
    range of compilers and languages than we assume for RandBLAS.

  * test_sample_indices.cc includes statistical tests for sampling from an index set with or without
    replacement. 

  * rng_common.hh includes data for statistical tables (e.g., for Kolmogorov-Smirnov tests) and helper
    functions to compute quantities associated with certain probability distributions (e.g., mean
    and variance of the hypergeometric distribution).


# OLD


Right-multplication by a structured linear operator in a GEMM-like API can
always be reduced to left-multiplication by flipping transposition flags and
layout parameters. So, why have equally fleshed-out tests(/test tooling) for
both cases?

Short answer: maybe it was a bad idea.

Big picture defensive answer:

    Different linear operators vary in the extent to which code for their 
    action on the left can be reduced to their action on the right. Action
    on the right is equivalent to adjoint-action from the left.

    Someone who's adding a new linear operator might prefer to think mostly
    in terms of right-multiplication, and just have left-multiplication reduce
    to adjoint-action from the right. 

    We want someone who adds new functionality to benefit from our testing infrastructure. So we made infrastructure to test GEMM-like APIs where
    one operand is structured, and it's easy to get started using this 
    infrastructure because it's equally valid to start with tests that
    multiply only for one side and only another.

Specifics:

    RSKGE3 doesn't actually reduce to LSKGE3. It could, but it was
    easy enough to have it reduce directly to GEMM, and reducing 
    directly to GEMM had the advantage of improved readibility. We don't
    test all possible combinations of flags (we omit when both arguments
    are transposed) but the combination we leave untested are unrelated
    to flow-of-control.

    RSKGES reduces to right_spmm, which does indeed fall back on
    left_spmm. But left_spmm has a large number codepaths (twelve!).
    It would have been awkward to write tests that hit all of those codepaths
    directly. Instead, we write a smaller set of tests for left_spmm
    and right_spmm, and count on the right_spmm tests to hit complementary
    codepaths compared to the paths hit in the left_spmm tests.
    (Admission: we don't know for sure if all codepaths are hit. Verifying
     that is on our TODO list.)
