import numpy as np
from sklearn.decomposition import PCA
from utils import fvecs_read, fvecs_write
import os
import struct

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['gist', 'sift']

if __name__ == "__main__":
    for dataset in datasets:
        print(f"PCA - {dataset}")
        # path
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        learn_path = os.path.join(path, f'{dataset}_learn.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        # read data vectors
        base = fvecs_read(base_path)
        learn = fvecs_read(learn_path)
        query = fvecs_read(query_path)

        D = learn.shape[1]
        pca_dim = D
        # projection
        pca = PCA(n_components=pca_dim)
        pca.fit(base)
        # save the transpose matrix

        projection_matrix = pca.components_.T
        base = np.dot(base, projection_matrix)
        learn = np.dot(learn, projection_matrix)
        query = np.dot(query, projection_matrix)

        save_base_path = os.path.join(path, f'{dataset}_base_pca_{pca_dim}.fvecs')
        save_learn_path = os.path.join(path, f'{dataset}_learn_pca_{pca_dim}.fvecs')
        save_query_path = os.path.join(path, f'{dataset}_query_pca_{pca_dim}.fvecs')
        matrix_save_path = f'./DATA/{dataset}_pca_matrix_{pca_dim}.fvecs'

        fvecs_write(matrix_save_path, projection_matrix)
        fvecs_write(save_base_path, base)
        fvecs_write(save_learn_path, learn)
        fvecs_write(save_query_path, query)
