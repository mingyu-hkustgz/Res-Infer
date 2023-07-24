import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['gist']

if __name__ == "__main__":

    for dataset in datasets:
        print(f"visual - {dataset}")
        original_data = fvecs_read(f'./logger/{dataset}_logger_OPQ60_ivf.fvecs')
        acc_dist = original_data[:, 0]
        app_dist = original_data[:, 1]
        cluster_dist = original_data[:, 2]
        thresh_dist = original_data[:, 3]
        num = min(500, thresh_dist.shape[0])
        tag = np.zeros(num)
        for i in tqdm(range(num)):
            if acc_dist[i] > thresh_dist[i]:
                tag[i] = 1
        feature_max = np.max(acc_dist[:num])
        feature_min = np.min(acc_dist[:num])
        x = np.linspace(feature_min, feature_max, 50)
        y = x
        for i in tqdm(range(num)):
            if tag[i] == 1:
                plt.scatter(app_dist[i] - cluster_dist[i], thresh_dist[i], c='b', alpha=0.1)
            else:
                plt.scatter(app_dist[i] - cluster_dist[i], thresh_dist[i], c='r', alpha=0.5)

        plt.plot(x, y, color='green', linewidth=1.0, linestyle='--')
        plt.show()
