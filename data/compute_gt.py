from utils import fvecs_read
from utils import fvecs_write, ivecs_read, ivecs_write
import os
import numpy as np
import faiss

source = '/home/yming/DATA/vector_data'
datasets = ['sift100m']


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
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        base = fvecs_read(base_path)
        learn = fvecs_read(query_path)
        # learn_num = int(1e5)
        # learn_data = learn[:learn_num]
        gt = do_compute_gt(base, learn, topk=100)
        save_path = os.path.join(source, f'{dataset}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_ground_path = os.path.join(save_path, f'{dataset}_groundtruth.ivecs')
        ivecs_write(save_ground_path, gt)
