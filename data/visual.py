import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['deep1M']
method_type = "OPQ"
method_dim = "128"
index_type = "ivf"
verbose = True
if __name__ == "__main__":

    for dataset in datasets:
        print(f"visual - {dataset}")
        filename = f'./logger/{dataset}_logger_{method_type}_{method_dim}_{index_type}.fvecs'
        original_data = fvecs_read(filename)
        acc_dist = original_data[:, 0]
        sub_dim = original_data.shape[1] - 2
        thresh_dist = original_data[:, -1]
        for model_id in range(1, sub_dim + 1):
            app_dist = original_data[:, model_id]
            num = min(10000, thresh_dist.shape[0])
            tag = np.zeros(num)
            class1_dist, class1_thresh = [], []
            class2_dist, class2_thresh = [], []
            for i in tqdm(range(num)):
                if acc_dist[i] > thresh_dist[i]:
                    class1_dist.append(app_dist[i])
                    class1_thresh.append(thresh_dist[i])
                else:
                    class2_dist.append(app_dist[i])
                    class2_thresh.append(thresh_dist[i])

            feature_max = np.max(acc_dist[:num])
            feature_min = np.min(thresh_dist[:num])
            x = np.linspace(feature_min, feature_min + 1, 50)
            y = x
            if verbose:
                plt.scatter(class1_dist, class1_thresh, c='b', label="acc > thresh", alpha=0.1)
                plt.scatter(class2_dist, class2_thresh, c='r', label="acc < thresh", alpha=0.1)
                plt.legend()
                plt.savefig(f'./figure/{dataset}/{dataset}_{method_type}_{method_dim}_{index_type}_{model_id}.png',
                            dpi=500)
                # plt.plot(x, y, color='green', linewidth=1.0, linestyle='--',alpha=0.6)
                plt.show()
