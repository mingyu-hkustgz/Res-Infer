# Effective and General Distance Computation for Approximate Nearest Neighbor Search (ICDE 2025)
## Prerequisites
**C++ require:**
* Eigen 
* Boost
* OpenMP

**python environment:**
* numpy
* faiss
* numpy
* scikit-learn
* matplotlib
* scipy
* tqdm

---
## Data set
* We recommend start from data set with learning data (GIST and DEEP)
* For dataset do noy provide learn data please refer ./data/data_split.py to split learning data from base
* The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html. 

## Reproduction
1. use ./data/compute_gt.py compute the learning query groundtruth
2. set the **store_path** and dataset in set.sh
3. run ```bash run.sh```
## Hardware Notice
* We have implemented an experimental environment under different hardware acceleration. 
* Specifically, if you use SIMD-AVX, please set the definition of Cmakelist to 
```
-std=c++17 -Ofast -march=core-avx2 -mavx512f -fpic -fopenmp -ftree-vectorize -fexceptions
```
* To disable SIMD, we add an executable target with different settings, please set the definition of Cmakelist to
```
-std=c++17 -O3
```
and comment out the corresponding executable target such as "search_ivf_512" or "search_hnsw_512".
## Baseline Notice
* We recommend that you test it under the corresponding settings, i.e. without SIMD acceleration - ADsampling, with SIMD acceleration - FINGER.
* The FINGER implemented in this project is mainly referenced by https://github.com/CaucherWang/Fudist and is still under review. Please use the original code https://github.com/Patrick-H-Chen/FINGER for evaluation (The comparison in our paper).
## Notice
1. The code is forked from https://github.com/gaoj0017/ADSampling we add multiprocess for fast index

## Reference
Reference to cite when you use this paper or code in a research paper:
```
@article{Yang2024EffectiveAG,
  title={Effective and General Distance Computation for Approximate Nearest Neighbor Search},
  author={Mingyu Yang and Wentao Li and Jiabao Jin and Xiaoyao Zhong and Xiangyu Wang and Zhitao Shen and Wei Jia and Wei Wang},
  year={2024},
  url={https://arxiv.org/abs/2404.16322}
}
```
