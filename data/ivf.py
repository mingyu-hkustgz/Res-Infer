import numpy as np
import faiss
import struct
import os
from utils import fvecs_write, fvecs_read
import argparse
source = '/home/DATA/vector_data'
# the number of clusters
K = 4096

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-m', '--method', help='approximate method', default='opq')
    parser.add_argument('-p', '--project', help='project dim', default=32)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    method = args['method']
    proj_dim = int(args['project'])
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
    X = fvecs_read(data_path)
    D = X.shape[1]
    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    fvecs_write(centroids_path, centroids)
    print(f"Clustering - {dataset}")
    if method == 'O':
        modified_cluster_path = f"./DATA/{dataset}/O{dataset}_centroid_{K}.fvecs"
        transformation_path = f'./DATA/{dataset}/O.fvecs'
    else:
        modified_cluster_path = f"./DATA/{dataset}/{dataset}_centroid_{method}_{proj_dim}.fvecs"
        transformation_path = f"./DATA/{dataset}/{dataset}_{method}_matrix_{proj_dim}.fvecs"

    P = fvecs_read(transformation_path)
    # modified_ centroids
    modified_centroids = np.dot(centroids, P)
    fvecs_write(modified_cluster_path, modified_centroids)
