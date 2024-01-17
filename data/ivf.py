import numpy as np
import faiss
import struct
import os
from utils import fvecs_write, fvecs_read
import argparse
source = '/home/DATA/vector_data'
pre_source = './DATA'
# the number of clusters
K = 4096

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-m', '--method', help='approximate method', default='pca')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    method = args['method']
    source = os.getenv('store_path')
    print(source)
    print(f"Clustering - {dataset}")
    if method == "naive":
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
    elif method == 'O':
        data_path = f'./DATA/{dataset}/O{dataset}_base.fvecs'
        centroids_path = f"./DATA/{dataset}/O{dataset}_centroid_{K}.fvecs"
        X = fvecs_read(data_path)
        D = X.shape[1]
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)
    elif method == 'pca':
        data_path = f'./DATA/{dataset}/{dataset}_base_{method}.fvecs'
        centroids_path = f"./DATA/{dataset}/{dataset}_centroid_{method}.fvecs"
        X = fvecs_read(data_path)
        D = X.shape[1]
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)
    elif method == 'opq':
        data_path = f'./DATA/{dataset}/{dataset}_base_{method}.fvecs'
        centroids_path = f"./DATA/{dataset}/{dataset}_centroid_{method}.fvecs"
        X = fvecs_read(data_path)
        D = X.shape[1]
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        fvecs_write(centroids_path, centroids)
