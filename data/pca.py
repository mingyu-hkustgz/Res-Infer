import numpy as np
from sklearn.decomposition import PCA
from utils import fvecs_read, fvecs_write
import os
import struct

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['gist', 'sift']

pca_dim = 10

if __name__ == "__main__":
    for dataset in datasets:
        print(f"PCA - {dataset}")
        # path
        path = os.path.join(source, dataset)
        learn_path = os.path.join(path, f'{dataset}_learn.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        save_learn_path = os.path.join(path, f'{dataset}_learn_pca_{pca_dim}.fvecs')
        save_query_path = os.path.join(path, f'{dataset}_query_pca_{pca_dim}.fvecs')
        matrix_save_path = f'./DATA/{dataset}_pca_{pca_dim}_matrix.fvecs'
        mean_save_path = f'./DATA/{dataset}_pca_{pca_dim}_mean.fvecs'
        # read data vectors
        learn = fvecs_read(learn_path)
        query = fvecs_read(query_path)
        D = learn.shape[1]
        # projection
        pca = PCA(n_components=pca_dim)
        pca.fit(learn)
        learn = pca.transform(learn)
        query = pca.transform(query)
        # save the transpose matrix
        projection_matrix = pca.components_.T
        fvecs_write(matrix_save_path, projection_matrix)
        pca_mean = pca.mean_[:, np.newaxis]
        fvecs_write(mean_save_path, pca_mean)
        # fvecs_write(save_learn_path, learn)
        # fvecs_write(save_query_path, query)
