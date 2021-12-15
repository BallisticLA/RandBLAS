# Some codes for sketching of structured matrices

In this folder I'm putting some codes for sketching of structured matrices. 
Structured sketches can roughly be categorized similarly to how the non-structured/standard sketches are classified:
- **Random dense sketches:** 
These are sketches made up of smaller components with each component chosen to be dense random matrices or tensors. 
Relevant paper references: 
[[BBB15](https://doi.org/10.1016/j.jcp.2014.10.009)]
[[SGTU18](https://r2learning.github.io/assets/papers/CameraReadySubmission%2041.pdf)] 
[[RR20](https://proceedings.mlr.press/v108/rakhshan20a.html)] 
[[RR21](https://tensorworkshop.github.io/NeurIPS2021/accepted_papers/Random_projections____Workshop_Neurips_2021%20(1).pdf)]

- **Structured SRTT:** 
These are similar to the standard SRTT sketches, but with added structure. 
Typically, the trigonometric transform and the diagonal sign matrix have added structure.
Relevant paper references: 
[[BBK18](https://doi.org/10.1137/17M1112303)] 
[[JKW20](https://doi.org/10.1093/imaiai/iaaa028)] 
[[MB20](https://doi.org/10.1016/j.laa.2020.05.004)] 
[[BKW21](https://arxiv.org/abs/2106.13349)] 
[[INRZ20](https://arxiv.org/abs/1912.08294)]

- **Hashing based:** 
TensorSketch is a hashing based sketch which is similar to CountSketch and is even faster for Kronecker structured vectors/matrices.
Relevant paper references: 
[[Pag13](https://doi.org/10.1145/2493252.2493254)]
[[PP13](https://doi.org/10.1145/2487575.2487591)] 
[[ANW14](https://papers.nips.cc/paper/2014/hash/b571ecea16a9824023ee1af16897a582-Abstract.html)] 
[[DSSW18](https://arxiv.org/abs/1712.09473)]

- **Recursive sketch:** 
This is a kind of sketch which combines sparse, SRTT and hashing based sketches in a recursive fashion.
Relevant paper reference: 
[[AKKP+20](https://arxiv.org/abs/1909.01410)]

- **Sampling based:** 
Unlike all the other sketches above this is a *data dependent* sketch, just like standard sampling based methods.
They are usually developed for specific applications, for example in alternating least squares algorithms for tensor decomposition.
Relevant paper references: 
[[CPLP16](https://proceedings.neurips.cc/paper/2016/hash/f4f6dce2f3a0f9dada0c2b5b66452017-Abstract.html)]
[[DJSSW19](https://arxiv.org/abs/1909.13384)]
[[LK20](https://arxiv.org/abs/2006.16438)] 
[[MB20](https://proceedings.mlr.press/v139/malik21b.html)] 
[[FGF21](https://arxiv.org/abs/2107.10654)] 
[[Mal21](https://arxiv.org/abs/2110.07631)]

## Subfolders

I have copied in existing code for structured matrix sketching in the following subfolders.
I may add additional folders later on.

- **kronecker-sketching:** 
This folder is a clone of the repo at https://github.com/OsmanMalik/kronecker-sketching. 
The repo contains code for recreating some experiments that appeared in [[MB20](https://doi.org/10.1016/j.laa.2020.05.004)] that compare different structured sketches.
The structured sketching methods included are:
    - The Tensor Random Projection (TRP) proposed in [[BBB15](https://doi.org/10.1016/j.jcp.2014.10.009)] and [[SGTU18](https://r2learning.github.io/assets/papers/CameraReadySubmission%2041.pdf)] which is a kind of random dense sketch.
    - A variant of the Kronecker fast JL transform (KFJLT) first proposed in [[BBK18](https://doi.org/10.1137/17M1112303)] which is a structured SRTT.
    - TensorSketch which is hashing based.
    - A sampling-based data-aware sketch which estimates leverage scores following a strategy first proposed in [[CPLP16](https://proceedings.neurips.cc/paper/2016/hash/f4f6dce2f3a0f9dada0c2b5b66452017-Abstract.html)].

The repo at https://github.com/OsmanMalik/TD-ALS-ES uses the variant of the recursive sketch [[AKKP+20](https://arxiv.org/abs/1909.01410)] that combines CountSketch and TensorSketch for tensor decomposition.
I didn't include those codes here since they are tailored for the specific problems that arise in the alternating least squares algorithm for CP and tensor ring decomposition, so they're not as clean as the codes in the repo above.
