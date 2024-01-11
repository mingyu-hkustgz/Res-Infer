import numpy as np
from sklearn.decomposition import PCA
from utils import fvecs_read, fvecs_write, ivecs_read, ivecs_write
import os
import argparse
import struct

source = '/home/BLD/mingyu/DATA/vector_data'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    print(f"PCA - {dataset}")
    # path
    path = os.path.join(source, dataset)
    base_path = os.path.join(path, f'{dataset}_base.fvecs')
    ground_path = os.path.join(path, f'{dataset}_learn_groundtruth.ivecs')
    # read data vectors
    base = fvecs_read(base_path)
    N, D = base.shape
    pca_dim = D
    # projection
    mean = np.mean(base, axis=0)
    base -= mean
    pca = PCA(n_components=pca_dim)
    pca.fit(base)
    # save the transpose matrix
    projection_matrix = pca.components_.T
    base = np.dot(base, projection_matrix)
    print(f"PCA - finished")
    save_base_path = f'./DATA/{dataset}/{dataset}_base_pca.fvecs'
    matrix_save_path = f'./DATA/{dataset}/{dataset}_pca_matrix.fvecs'

    variance = np.var(base, axis=0)
    save_matrix = np.vstack((mean, mean, variance, projection_matrix))
    fvecs_write(matrix_save_path, save_matrix)
    fvecs_write(save_base_path, base)

    for K in [20, 100]:
        matrix_save_path = f'./DATA/{dataset}/{dataset}_pca_matrix_{K}.fvecs'
        ground = ivecs_read(ground_path)
        ground = ground[:, :K]
        ground = ground.flatten()
        X_sample = base[ground]
        sample_mean = np.mean(X_sample, axis=0)
        X_sample -= sample_mean
        variance = np.var(X_sample, axis=0)
        save_matrix = np.vstack((mean, sample_mean, variance, projection_matrix))
        fvecs_write(matrix_save_path, save_matrix)
