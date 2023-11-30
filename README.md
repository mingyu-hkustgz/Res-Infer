
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

## Notice
1. The code is fork from https://github.com/gaoj0017/ADSampling we add multiprocess for fast index