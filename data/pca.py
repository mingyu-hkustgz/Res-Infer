import numpy as np
from sklearn.decomposition import PCA
from utils import fvecs_read, fvecs_write
import os
import argparse
import struct

source = '/home/BLD/mingyu/DATA/vector_data'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-p', '--project', help='project dim', default=32)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    proj_dim = args['project']
    print(f"PCA - {dataset}")
    # path
    path = os.path.join(source, dataset)
    base_path = os.path.join(path, f'{dataset}_base.fvecs')
    # read data vectors
    base = fvecs_read(base_path)
    N, D = base.shape
    sample_count = min(N, 5000000)
    X_sample = base[:sample_count]
    pca_dim = D
    # projection
    mean = np.mean(X_sample, axis=0)
    pca = PCA(n_components=pca_dim)
    pca.fit(X_sample)
    # save the transpose matrix
    base -= mean
    projection_matrix = pca.components_.T
    base = np.dot(base, projection_matrix)

    save_base_path = f'./DATA/{dataset}/{dataset}_base_pca_{proj_dim}.fvecs'
    matrix_save_path = f'./DATA/{dataset}/{dataset}_pca_matrix_{proj_dim}.fvecs'
    save_matrix = np.vstack((mean, projection_matrix))
    fvecs_write(matrix_save_path, save_matrix)
    fvecs_write(save_base_path, base)
