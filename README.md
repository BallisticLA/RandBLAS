# RandBLAS

RandBLAS is a C++ library for the most basic operation in randomized numerical linear algebra: sketching.
We want RandBLAS to eventually become a standard like the BLAS.
Our goal is for the code at this repository to merely be "Reference RandBLAS."

The library hosted here will surely undergo several major revisions before its API stabilizes.
Therefore we insist that no one use RandBLAS as a dependency in a larger project for the time being.
While RandBLAS is dependency for a library called ["RandLAPACK"](https://github.com/BallisticLA/proto_randlapack) which we are also developing, we are preparred to make major changes to RandLAPACK whenever a major change is made to RandBLAS.

Refer to ``INSTALL.md`` for directions on how to install RandBLAS' dependencies, install
RandBLAS itself, and use RandBLAS in other projects.


## The z_scratch/ directory 

This is a scratch workspace for debugging, preparing examples, and informal benchmarking.

If you work in this repo, create a folder with your name in ``z_scratch/``.
You have total authority over what goes in that folder.
If you have something presentable that you want to reference later, then a copy should be kept in the "sharing" folder.

When you make an entry in the sharing folder you should has the format "year-mm-dd-[your initials]-[letter]".
 * You substitute-in "[your initials]" with whatever short character string is appropriate for you (for Riley John Murray, that's rjm).
The "[letter]" should start every day with "a" and then increment along the alphabet.
 * Hopefully you won't have more than 26 folders you'd want to add to "sharing" on one day.

If you need source code for a third-party library in order to run your experiments, add a git submodule.
