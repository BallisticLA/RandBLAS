Procedures for making a new RandBLAS release
============================================

This file provides the procedures for releasing a new version of RandBLAS. The process involves defining a new release in the commit history, changing the default version that CMake reports if it can't detect Git, and deploying updated web documentation.

# Defining a new release

I think this just requires making a git tag. Tags of the form X.Y.Z should work out of the box.
Tags with prerelease info (like alpha, beta, etc..) might require reworking the string 
parsing in RandBLAS/CMake/rb_version.cmake.

This ends up being simpler than CVXPY's method of defining new releases, which requires data
specified manually in cvxpy/setup/versioning.py, while we infer that data from a tag with the
help of ``git describe``. 

# Writing release notes

This is self-explanatory.

# Updating web docs

## Updating web doc sources

Appropriately adapt the release notes. Maybe just link to the GitHub releases for now.

## ReadTheDocs deployment

Change the default version of the web docs that people see.
Maybe have to hard-code the version for a given branch.

# Creating a new release on GitHub

This is self-explanatory.
