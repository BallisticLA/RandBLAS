Developer notes for RandBLAS' testing infrastructure
====================================================

Meta note:

    Don't defend previous design decisions. Just explain how they work. 
    That's easier and more useful. Plus, a good explanation will make
    pros and cons of the decision (more) self-evident.




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
    are transposed) but the combinationed we leave untested are unrelated
    to flow-of-control.

    RSKGES reduces to right_spmm, which does indeed fall back on
    left_spmm. But left_spmm has a large number codepaths (twelve!).
    It would have been awkward to write tests that hit all of those codepaths
    directly. Instead, we write a smaller set of tests for left_spmm
    and right_spmm, and count on the right_spmm tests to hit complementary
    codepaths compared to the paths hit in the left_spmm tests.
    (Admission: we don't know for sure if all codepaths are hit. Verifying
     that is on our TODO list.)
