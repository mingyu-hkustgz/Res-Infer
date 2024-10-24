import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import fvecs_read
import argparse
from tqdm import tqdm

source = '/home/yming/DATA/vector_data'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    parser.add_argument('-m', '--method', help='approximate method', default='pca')
    parser.add_argument('-i', '--indextype', help='index type', default='hnsw1')
    parser.add_argument('-v', '--verbose', help='visual option', default=False)
    parser.add_argument('-k', '--K', help='K nearest neighbor', default=1)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    method_type = args['method']
    index_type = args['indextype']
    verbose = args['verbose']
    K = args['K']
    print(f"visual - {dataset}")
    filename = f'./logger/{dataset}_logger_{method_type}_{index_type}.fvecs'
    save_path = f'./DATA/{dataset}/linear/linear_{index_type}_{method_type}_{K}.log'
    original_data = fvecs_read(filename)
    acc_dist = original_data[:, 0]
    cluster_dist = original_data[:, 0]
    thresh_dist = original_data[:, -1]
    if method_type == "opq":
        cluster_dist = original_data[:, -2]
        model_num = 1
    else:
        model_num = original_data.shape[1] - 2
    W_ = []
    B_ = []
    for model_id in range(1, model_num + 1):
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("parameters：", model.coef_, model.intercept_)

        x_boundary = np.linspace(X[:10000, 0].min(), X[:10000, 0].max(), 100)
        y_boundary = -(model.coef_[0][0] * x_boundary + model.intercept_) / model.coef_[0][1]
        if verbose:
            plt.scatter(X[:5000000:1000, 0], X[:5000000:1000, 1], c=y[:5000000:1000], alpha=0.2)
            plt.plot(x_boundary, y_boundary, "r--")
            plt.xlabel("approximate dist")
            plt.ylabel("threshold dist")
            plt.title("binary classifier")
            plt.savefig(f'./figure/{dataset}/{dataset}_{method_type}_linear_{index_type}_{model_id}.png', dpi=500)

            plt.show()

        print("report：")
        print(classification_report(y_test, y_pred))

        if method_type == "pca":
            w1 = model.coef_[0][0]
            w2 = model.coef_[0][1]
            b = model.intercept_[0]
            w1 /= -w2
            b /= -w2
            W_.append(w1)
            B_.append(b)
        else:
            w1 = model.coef_[0][0]
            w2 = model.coef_[0][1]
            w3 = model.coef_[0][2]
            b = model.intercept_[0]
            w1 /= -w2
            w3 /= -w2
            b /= -w2
            W_.append(w1)
            W_.append(w3)
            B_.append(b)

    if method_type == "opq":
        f = open(save_path, 'w')
        print(1, file=f)
        print("%.6f %.6f %.6f" % (W_[0], W_[1], B_[0]), file=f)
        f.close()
    else:
        f = open(save_path, 'w')
        print(len(W_), file=f)
        for i in range(len(W_)):
            print("%.6f %.6f %.6f" % (W_[i], B_[i], B_[i]), file=f)
        f.close()
