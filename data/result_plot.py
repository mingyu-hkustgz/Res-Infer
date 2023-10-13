import numpy as np
import matplotlib.pyplot as plt
from utils import fvecs_read, fvecs_write
import os
import struct
from tqdm import tqdm

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['gist', 'deep1M', '_msong', '_tiny5m', '_glove2.2m', '_word2vec']

hnsw_marker = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
ivf_marker = ['s', 's', 's', 's', 's', 's', 's', 's', 's']
col = ['r', 'b', 'y', 'g', 'c', 'm', 'olive', 'gold', 'violet']


def load_result_data(filename):
    f = open(filename)
    tag0, tag1, tag2 = [], [], []
    line = f.readline()
    while line:
        raw = line.split(' ')
        tag0.append(float(raw[1]))
        tag1.append(1e6 / float(raw[2]))
        line = f.readline()
    f.close()
    return tag0, tag1


if __name__ == "__main__":

    for dataset in datasets:
        print(f"visual - {dataset}")
        for K in [20, 100]:
            plt.figure(figsize=(12, 8))
            file_path = f"./results/recall@{K}/{dataset}"
            if not os.path.exists(file_path):
                continue
            for i in range(9):
                result_path = f"./results/recall@{K}/{dataset}/{dataset}_ad_hnsw_{i}.log"
                if not os.path.exists(result_path):
                    continue
                label = "null"
                if i == 0:
                    label = "hnsw-naive"
                elif i == 1:
                    label = "hnsw++"
                elif i == 2:
                    label = "hnsw+"
                    continue
                elif i == 3:
                    label = "hnsw-opq+"
                elif i == 4:
                    label = "hnsw-opq++"
                elif i == 5:
                    label = "hnsw-opq+sse"
                elif i == 6:
                    label = "hnsw-opq++sse"
                elif i == 7:
                    label = "hnsw-pca+"
                elif i == 8:
                    label = "hnsw-pca++"

                recall, Qps = load_result_data(result_path)
                plt.plot(recall, Qps, marker=hnsw_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white')

            plt.xlabel("Recall@100")
            plt.ylabel("Qps")
            plt.legend(loc="upper right")
            plt.grid(linestyle='--', linewidth=0.5)
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
            plt.rc('font', family='Times New Roman')
            plt.savefig(f'./figure/{dataset}_hnsw_qps_@{K}.png', dpi=400)
            plt.show()
            plt.figure(figsize=(12, 8))
            for i in range(7):
                result_path = f"./results/recall@{K}/{dataset}/{dataset}_ad_ivf_{i}.log"
                label = "null"
                if not os.path.exists(result_path):
                    continue
                if i == 0:
                    label = "ivf-naive"
                elif i == 1:
                    label = "ivf++"
                elif i == 2:
                    label = "ivf+"
                elif i == 3:
                    label = "ivf-opq"
                elif i == 4:
                    label = "ivf-opq-sse"
                elif i == 5:
                    label = "ivf-pca++"
                elif i == 6:
                    label = "ivf-pca+"

                recall, Qps = load_result_data(result_path)
                plt.plot(recall, Qps, marker=ivf_marker[i], c=col[i], label=label, alpha=0.5, linestyle="--", markerfacecolor='white')

            plt.xlabel("Recall@100")
            plt.ylabel("Qps")
            plt.legend(loc="upper right")
            plt.grid(linestyle='--', linewidth=0.5)
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
            plt.rc('font', family='Times New Roman')
            plt.savefig(f'./figure/{dataset}_ivf_qps_@{K}.png', dpi=400)
            plt.show()
