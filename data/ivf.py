import numpy as np
import faiss
import struct
import os
from utils import fvecs_write, fvecs_read

source = '/home/DATA/vector_data'
datasets = ['gist']
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
        opq_cluster_path = os.path.join(path, f"{dataset}_centroid_opq_120.fvecs")
        pca_cluster_path = os.path.join(path, f"{dataset}_centroid_pca_960.fvecs")
        transformation_path = os.path.join(path, 'O.fvecs')
        opq_tran_path = f"./DATA/gist_opq_matrix_120.fvecs"
        pca_tran_path = f"./DATA/gist_pca_matrix_960.fvecs"
        # read data vectors
        X = fvecs_read(data_path)
        P = fvecs_read(transformation_path)
        P_opq = fvecs_read(opq_tran_path)
        P_pca = fvecs_read(pca_tran_path)

        D = X.shape[1]
        
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)

        # randomized centroids
        randomized_centroids = np.dot(centroids, P)
        opq_centroids = np.dot(centroids, P_opq)
        pca_centroids = np.dot(centroids, P_pca)
        fvecs_write(randomzized_cluster_path, randomized_centroids)
        fvecs_write(opq_cluster_path, opq_centroids)
        fvecs_write(pca_cluster_path, pca_centroids)




