from utils import fvecs_read
from utils import fvecs_write, ivecs_read, ivecs_write
import os
import numpy as np
import faiss

source = '/home/yming/DATA/vector_data'
datasets = ['sift10m','tiny5m','glove2.2m','word2vec']


def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.verbose = True
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')


if __name__ == "__main__":
    for dataset in datasets:
        print(f'current dataset: {dataset}')
        path = os.path.join(source, dataset)
        base_path = os.path.join(path, f'{dataset}_base.fvecs')
        origin_data = fvecs_read(base_path)
        np.random.shuffle(origin_data)
        origin_num = origin_data.shape[0]
        learn_num = int(1e4) - int(1e3)
        query_num = int(1e3)
        split_num = int(1e4 + 1e3)
        base_data = origin_data[:-split_num]
        learn_data = origin_data[-split_num:-query_num]
        query_data = origin_data[-query_num:]

        gt = do_compute_gt(base_data, query_data, topk=100)
        learn_gt = do_compute_gt(base_data, learn_data, topk=100)
        save_path = os.path.join(source, f'_{dataset}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_base_path = os.path.join(save_path, f'_{dataset}_base.fvecs')
        save_learn_path = os.path.join(save_path, f'_{dataset}_learn.fvecs')
        save_query_path = os.path.join(save_path, f'_{dataset}_query.fvecs')
        save_ground_path = os.path.join(save_path, f'_{dataset}_groundtruth.ivecs')
        save_learn_ground_path = os.path.join(save_path, f'_{dataset}_learn_groundtruth.ivecs')

        fvecs_write(save_base_path, base_data)
        fvecs_write(save_learn_path, learn_data)
        fvecs_write(save_query_path, query_data)
        ivecs_write(save_ground_path, gt)
        ivecs_write(save_learn_ground_path, learn_gt)
