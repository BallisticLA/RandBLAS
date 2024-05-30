## Getting sparse data matrices for the examples
This shell script shows how to get sparse matrices in from the SuiteSparse Matrix Collection:

  https://sparse.tamu.edu/.

Consider a page for a specific matrix in that collection:

  https://sparse.tamu.edu/Schulthess/N_reactome.

At time of writing, that page (and ones like it) have "download" section with three clickable buttons.
One of those buttons says "Matrix Market". If you right-click that button you can get the link that it
points to:

  https://suitesparse-collection-website.herokuapp.com/MM/Schulthess/N_reactome.tar.gz.

You can then download that file with a shell command like ``wget``, and uncompress it with a command
like ``tar``. See below for commands that would be executed when running a bash terminal on macOS.

  Note: you'll need to run these commands (or some like them) in order to call some of the 
  examples in ``RandBLAS/examples/sparse-low-rank-approx`` without specifying a Matrix Market file.

```shell
wget https://suitesparse-collection-website.herokuapp.com/MM/Schulthess/N_reactome.tar.gz
tar -xvzf N_reactome.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz
tar -xvzf bcsstk17.tar.gz
```

