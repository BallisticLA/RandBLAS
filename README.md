This repository hosts BALLISTIC-internal RBLAS prototyping.
This includes a proper C++ RBLAS prototype and scratch work
for very rough benchmarking.

The directory "install/", its subdirectories, and the directory "build/" should only have .git-keep files upon cloning.

The directory "z_scratch/" is a scratch workspace for debugging, preparing examples, and informal benchmarking.

## 1. RBLAS dependencies

RBLAS depends on BLAS++ and Random123.

### 1.1. Managing the BLAS++ dependency

* You have to compile and install BLAS++ from [source](https://bitbucket.org/icl/blaspp/src/master/).
    * The library name for BLAS++ is "blaspp".
    * The recommended installation directory is ``/opt/slate/blaspp``.
* Here's what happens with CMake in connecting to BLAS++:
    * We assume access to a CMake variable "BLASPP_DIR" that stores the location of "blaspp" library and header files.
      This variable needs to be set as an argument to CMake by a flag ``-DBLASPP_DIR=<insert location here>``.
      You can either set that flag manually each time you run CMake or you can use an IDE that will do that for you.
    * CMake will look for the blaspp library file in a subdirectory called ``lib`` within
      your BLASPP_DIR directory. That is, it will look in a place like ``opt/slate/blaspp/lib``, and then it will
      save the precise library file name to a variable called BLASPP (as below) 
      ```
      find_library(BLASPP blaspp /opt/slate/blaspp/lib)
      ```
    * CMake will look for the ``blas.hh`` header file within the ``include`` subdirectory of BLASPP_DIR.

### 1.2: Managing the Random123 dependency

* Random123 is header-only, so there's no separate "compile and install" step like BLAS++. 
* You need to download the [source files](https://github.com/DEShawResearch/random123) and put them somewhere
  special. We recommend creating a folder ``/opt/random123`` and populating it with the ``/include/Random123``
  directory at the top level of the Random123 GitHub Repo. So the files at 
  ```
  https://github.com/DEShawResearch/random123/tree/main/include/Random123
  ```
  should be at a place like ``/opt/random123/include/Random123``.
* CMake assumes acess to a variable "RANDOM123_DIR" which is one level above the Random123 header files.
  So with our recommended settings that would be ``/opt/random123/include``. This needs to be set as an
  argument to CMake by a flag ``-DRANDOM123_DIR=<insert location here>``. You can either set that manually
  each time you call CMake or you can use an IDE that will do that for you.

## 2. How to build RBLAS, its tests, and run the tests

This should be simple. First, get cmake to populate ``build/`` with the necessary Makefiles and other build
metadata. If you're using a properly configured IDE like VS Code you should be able to do that by typing
"ctrl + s" while having ``CMakeLists.txt`` open. If you're not using an IDE then you should change directory
into ``build/`` and then run cmake with something like 
```
cmake -DBLASPP_DIR=<your blaspp dir> -DRANDOM123_DIR=<your random123 dir> ..
```
You might also want to set C++ compiler flags like ``-O0 -g`` to enable easy debugging.
With those flags and our recommended locations for BLAS++ and Random123, the cmake command would be
```
cmake -DBLASPP_DIR=/opt/slate/blaspp -DRANDOM123_DIR=/opt/random123/include -DCXXFLAGS='-O0 -g' ..
```

The next step is to run ``make && make install``. This will combile rblas
and its tests, and then write the final files to appropriate places in 
``install/``.

You can run the tests by cd-ing back to the repo's outer-most directory and then running ``install/bin/./rblas_tests``.


## 3. The z_scratch/ directory 

If you work in this repo, create a folder with your name in ``z_scratch/``.
You have total authority over what goes in that folder.
If you have something presentable that you want to reference later, then a copy should be kept in the "sharing" folder.

When you make an entry in the sharing folder you should has the format "year-mm-dd-[your initials]-[letter]".
 * You substitute-in "[your initials]" with whatever short character string is appropriate for you (for Riley John Murray, that's rjm).
The "[letter]" should start every day with "a" and then increment along the alphabet.
 * Hopefully you won't have more than 26 folders you'd want to add to "sharing" on one day.

If you need source code for a third-party library in order to run your experiments, add a git submodule.
