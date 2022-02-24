This is a repo to host BALLISTIC-internal RBLAS prototyping.

Most directories at the top of this repo are for a CMake-organized C++ library called "rblas".

The folders "lib", "bin", and "include" should all be empty (or only contain .git-keep) upon cloning.
The directory "z_scratch/" is a scratch workspace for debugging, preparing examples, and informal benchmarking.

## RBLAS dependencies

RBLAS dependencies are BLAS++ and Random123.

Managing the BLAS++ dependency:
* You have to compile and install BLAS++ from [source](https://bitbucket.org/icl/blaspp/src/master/).
    * The library name for BLAS++ is "blaspp".
    * We assume the installation directory is ``/opt/slate/blaspp``.
* This project's CMakeLists files take three steps to connect with BLAS++:
    * Create a CMake variable called "BLASPP" that stores the location of the "blaspp" library,
      with a hint that CMake should look in ``opt/slate/blaspp/lib`` for that library: 
      ```
      find_library(BLASPP blaspp /opt/slate/blaspp/lib)
      ```
    * Specify ``/opt/slate/blaspp/include/blas.hh`` explicitly when defining a target library or executable.
    * Set ``/opt/slate/blaspp/include`` as an include directory for each relevant target.

Managing the Random123 dependency:
* Random123 is header-only, so there's no separate "compile and install" step like BLAS++. 
* You need to download the [source files](https://github.com/DEShawResearch/random123) and put them somewhere
  special. We assume the contents of 
  ```
  https://github.com/DEShawResearch/random123/tree/main/include/Random123
  ```
  are stored in ``/opt/random123/include/Random123``. Nothing else needs to go in ``/opt/random123``.


## The z_scratch/ directory 

If you work in this repo, create a folder with your name in ``z_scratch/``.
You have total authority over what goes in that folder.
If you have something presentable that you want to reference later, then a copy should be kept in the "sharing" folder.

When you make an entry in the sharing folder you should has the format "year-mm-dd-[your initials]-[letter]".
 * You substitute-in "[your initials]" with whatever short character string is appropriate for you (for Riley John Murray, that's rjm).
The "[letter]" should start every day with "a" and then increment along the alphabet.
 * Hopefully you won't have more than 26 folders you'd want to add to "sharing" on one day.

If you need source code for a third-party library in order to run your experiments, put the source code
in ``z_scratch/z_extern``.
