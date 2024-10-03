import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import fvecs_read
import argparse
from scipy.stats import norm
from tqdm import tqdm

source = '/home/yming/DATA/vector_data'


def load_multi_model(file_name):
    f = open(file_name, "r")
    line = f.readline()
    line = f.readline()
    W, B = [], []
    while line:
        line = line.strip('\n')
        line_tuple = line.split(" ")
        W.append(float(line_tuple[0]))
        B.append(float(line_tuple[1]))
        line = f.readline()
    f.close()
    return W, B


def load_single_model(file_name):
    f = open(file_name, "r")
    line = f.readline()
    line = f.readline()
    W, B = [], []
    while line:
        line = line.strip('\n')
        line_tuple = line.split(" ")
        W.append(float(line_tuple[0]))
        W.append(float(line_tuple[1]))
        B.append(float(line_tuple[2]))
        line = f.readline()
    f.close()
    return W, B


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    parser.add_argument('-m', '--method', help='approximate method', default='pca')
    parser.add_argument('-p', '--projdim', help='project dim', default='32')
    parser.add_argument('-i', '--indextype', help='index type', default='hnsw1')
    parser.add_argument('-v', '--verbose', help='visual option', default=False)
    parser.add_argument('-k', '--K', help='K nearest neighbor', default=1)
    args = vars(parser.parse_args())
    dataset = "deep1M"
    method_type = "pca"
    method_dim = "32"
    index_type = "hnsw1"
    verbose = True
    K = 20
    print(f"visual - {dataset}")
    filename = f'./logger/{dataset}_logger_{method_type}_{method_dim}_{index_type}.fvecs'
    # linear_path = f'./DATA/{dataset}/linear/linear_hnsw1_{method_type}_{method_dim}_{K}.log'
    original_data = fvecs_read(filename)
    acc_dist = original_data[:, 0]
    cluster_dist = original_data[:, 0]
    thresh_dist = original_data[:, -1]


    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 18})
    for model_id in [2, 3, 4]:
        app_dist = original_data[:, model_id]
        num = min(5000000, app_dist.shape[0])
        y = np.zeros(num, dtype=int)
        if method_type == "opq":
            X = np.zeros((num, 3), dtype=float)
        else:
            X = np.zeros((num, 2), dtype=float)

        for i in tqdm(range(num)):
            if acc_dist[i] > thresh_dist[i]:
                y[i] = 1
                X[i][0] = app_dist[i]
                X[i][1] = thresh_dist[i]

        X_sample = []
        for i in range(num):
            if y[i] == 1:
                X_sample.append(X[i][0])

        X_sample.sort()
        dim = (model_id-1) * 32
        arr_mean = np.mean(X_sample)
        arr_var = np.var(X_sample)
        print(arr_mean, arr_var)
        print("%.8f" % arr_var)
        # print(X_sample)
        plt.hist(X_sample, bins=100, alpha=0.2,label=f"proj dim= {dim}")
        print(arr_mean)

    plt.legend()
    plt.show()
    plt.savefig('./figure/bias.jpg')
