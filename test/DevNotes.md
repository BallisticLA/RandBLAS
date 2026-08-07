# Developer notes for RandBLAS' testing infrastructure

This document doesn't defend previous design decisions.
It just explains how things work right now.
That's easier for me (Riley) to write, and it's more useful to others.
(Plus, it helps make the pros and cons of the current approach self-evident.)

Nothing defined in this folder is part of RandBLAS' public API.

## Contents

### linops

Relies on RandBLAS/testing/linops.hh and test/linops/linop_common.hh.

Tests of "meaty" functions.

  * lskges, rskges, lskge3, rskge3. The rskgex (x=3 or x=s) functions could reduce to lskgex by transposing the
    product and flipping the layout. Strictly speaking, the rskgex functions don't do that, but they
    easily could. In any case, we currently have similar tests for rskgex and lskgex.
  * left_spmm and right_spmm. The right_spmm implementation falls back on left_spmm. Despite this,
    right_spmm has its own set of tests.

Tests of wrapper functions.

  * sketch_vector. It reduces to the same sketch_general no matter the type of the sketching operator.
  * sketch_sparse. It reduces to left_spmm/right_spmm no matter the type of the data matrix. (The
    sketching operator type is naturally fixed to DenseSkOp.)
  * sketch_symmetric. It reduces to the same sketch_general no matter the type of the sketching operator.


### basic_rng

Relies on RandBLAS/testing/stats.hh.

  * `test_philox.cc` validates the native Philox implementation against 204
    static vectors generated offline from a pinned Random123 revision. The test
    suite never locates or executes Random123.
  * `test_word_array.cc` covers modular carry propagation and wraparound.
  * `test_rng_state.cc` checks engine/state concepts, scalar seed mapping,
    output-only block generation, explicit advancement, const raw accessors,
    and compatibility with an engine whose counter representation is opaque.
  * `test_repacked_output.cc` checks direct and nested repacking, including
    least-significant-chunk-first ordering.
  * `test_distributions.cc` checks the native integer-to-floating transforms,
    Box--Muller reference values, endpoints, and word assignment.
  * `test_sampler_regression.cc` protects the pre-migration dense and sparse
    streams. Sparse outputs are bitwise exact; dense comparisons use the narrow
    floating-point tolerance required for host math-library differences.

  * test_discrete.cc includes statistical tests for sampling from an index set with or without
    replacement.

Sampler tests elsewhere cover state advancement, full/submatrix agreement, and
OpenMP thread-count independence. Downstream-package and example builds verify
that no external random-number package is required.

### RNG stream

`RandBLAS/testing/rng.hh` contains the test-only `detail::RNGStream` adapter.
It turns fixed result blocks into a sequential word stream for random sparse
test-matrix generation and supplies the uniform, Gaussian, and geometric draws
needed there. Loading a block advances its held state immediately; unread lanes
remain in its local buffer. Production RandBLAS sampling remains
coordinate-addressed and does not use this sequential adapter.


# OLD

Right-multiplication by a structured linear operator in a GEMM-like API can
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

    We want someone who adds new functionality to benefit from our testing infrastructure.
    So we made infrastructure to test GEMM-like APIs where
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
