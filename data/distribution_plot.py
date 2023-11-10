import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import fvecs_read
import argparse
from scipy.stats import norm
from tqdm import tqdm

source = '/home/BLD/mingyu/DATA/vector_data'


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
    dataset = "gist"
    method_type = "opq"
    method_dim = "120"
    index_type = "hnsw1"
    verbose = True
    K = 100
    print(f"visual - {dataset}")
    filename = f'./logger/{dataset}_logger_{method_type}_{method_dim}_{index_type}.fvecs'
    linear_path = f'./DATA/{dataset}/linear/linear_hnsw1_{method_type}_{method_dim}_{K}.log'
    original_data = fvecs_read(filename)
    acc_dist = original_data[:, 0]
    cluster_dist = original_data[:, 0]
    thresh_dist = original_data[:, -1]
    if method_type == "opq":
        cluster_dist = original_data[:, -2]
        model_num = 1
        W, B = load_single_model(linear_path)
        print(W)
    else:
        model_num = original_data.shape[1] - 2
        W, B = load_multi_model(linear_path)
        print(W)

    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(5, 4))
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.axis('off')
    for model_id in [1]:
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
            if method_type == "opq":
                X[i][0] = app_dist[i]
                X[i][1] = thresh_dist[i]
                X[i][2] = cluster_dist[i]
            else:
                X[i][0] = app_dist[i]
                X[i][1] = thresh_dist[i]

        X_sample = []
        for i in range(num):
            if y[i] == 1:
                if method_type == "opq":
                    X_sample.append(X[i][0] * W[0] + X[i][2] * W[1] + B[0] - acc_dist[i])
                else:
                    X_sample.append(X[i][0] * W[model_id - 1] + B[model_id - 1] - acc_dist[i])

        X_sample.sort()
        dim = model_id * 120
        arr_mean = np.mean(X_sample)
        arr_var = np.var(X_sample)
        pdf = norm.pdf(X_sample, arr_mean, arr_var)
        # print(X_sample)
        ax2.hist(X_sample, bins=500, alpha=0.5)
        print(arr_mean)
        ax1.plot(X_sample, pdf, label=f'dim={dim}', alpha=0.85)

    ax1.legend()
    plt.show()
