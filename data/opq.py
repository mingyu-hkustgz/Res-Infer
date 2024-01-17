import numpy as np
import faiss
import struct
import os
from utils import fvecs_read, fvecs_write
import argparse

source = '/home/BLD/mingyu/DATA/vector_data/'
M = 64
nbits = 8

def save_centroid(filename, data):
    print(f"Writing centroid file - {filename}")
    M, k, d = data.shape
    with open(filename, 'wb') as fp:
        item = struct.pack('I', M)
        fp.write(item)
        item = struct.pack('I', k)
        fp.write(item)
        item = struct.pack('I', d)
        fp.write(item)
        for x in data:
            for y in x:
                for z in y:
                    a = struct.pack('f', z)
                    fp.write(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    parser.add_argument('-b', '--bits', help='cluster bits', default=8)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    nbits = int(args['bits'])
    print(f"OPQ transform - {dataset}")
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    X_base = fvecs_read(data_path)
    d = X_base.shape[1]
    M = 64
    if dataset == "gist":
        M = int(d / 8)
    else:
        M = int(d / 4)
    d2 = ((d + M - 1) // M) * M
    opq = faiss.OPQMatrix(d, M, d2)
    opq.verbose = True
    opq.train(X_base)
    Matrix_A = faiss.vector_float_to_array(opq.A)
    Matrix_A = Matrix_A.reshape(d2, d2)
    # save the transpose matrix
    fvecs_write(f'./DATA/{dataset}/{dataset}_opq_matrix.fvecs', Matrix_A.T)
    X_base = opq.apply(X_base)
    pq = faiss.ProductQuantizer(d2, M, nbits)
    pq.verbose = True
    pq.train(X_base)
    centroids = faiss.vector_float_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
    save_centroid(f'./DATA/{dataset}/{dataset}_codebook.centroid', centroids)
    fvecs_write(f'./DATA/{dataset}/{dataset}_base_opq.fvecs', X_base)
