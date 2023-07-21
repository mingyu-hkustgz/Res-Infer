import numpy as np
import faiss
import struct
import os
from utils import fvecs_write, fvecs_read

source = '/home/DATA/vector_data'
datasets = ['gist', 'sift']
# the number of clusters
K = 4096 


if __name__ == '__main__':

    for dataset in datasets:
        print(f"Clustering - {dataset}")
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        randomzized_cluster_path = os.path.join(path, f"O{dataset}_centroid_{K}.fvecs")
        transformation_path = os.path.join(path, 'O.fvecs')

        # read data vectors
        X = fvecs_read(data_path)
        P = fvecs_read(transformation_path)

        D = X.shape[1]
        
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)

        # randomized centroids
        centroids = np.dot(centroids, P)
        fvecs_write(randomzized_cluster_path, centroids)
