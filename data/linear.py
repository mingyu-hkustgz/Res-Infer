import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import fvecs_read
import argparse
from tqdm import tqdm

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['deep1M']
method_type = "OPQ"
method_dim = "64"
index_type = "ivf"
verbose = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='linear regression')
    parser.add_argument('-d', '--dataset', help='dataset', default='deep1M')
    parser.add_argument('-m', '--method', help='approximate method', default='OPQ')
    parser.add_argument('-p', '--projdim', help='project dim', default='64')
    parser.add_argument('-i', '--indextype', help='index type', default='ivf',)
    parser.add_argument('-v', '--verbose', help='visual option', default=False)

    args = vars(parser.parse_args())
    dataset = args['dataset']
    method_type = args['method']
    method_dim = args['projdim']
    index_type = args['indextype']
    verbose = args['verbose']

    print(f"visual - {dataset}")
    filename = f'./logger/{dataset}_logger_{method_type}_{method_dim}_{index_type}.fvecs'
    save_path = f'./DATA/{dataset}/linear_{index_type}_{method_type}_{method_dim}.log'
    original_data = fvecs_read(filename)
    acc_dist = original_data[:, 0]
    thresh_dist = original_data[:, -1]
    model_num = original_data.shape[1] - 2
    W_ = []
    B_ = []
    for model_id in range(1, model_num + 1):
        app_dist = original_data[:, model_id]
        num = min(1000000, app_dist.shape[0])
        y = np.zeros(num, dtype=int)
        X = np.zeros((num, 2), dtype=float)
        for i in tqdm(range(num)):
            if acc_dist[i] > thresh_dist[i]:
                y[i] = 1

            X[i][0] = app_dist[i]
            X[i][1] = thresh_dist[i]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("parameters：", model.coef_, model.intercept_)

        plt.scatter(X[:10000, 0], X[:10000, 1], c=y[:10000], alpha=0.2)
        x_boundary = np.linspace(X[:10000, 0].min(), X[:10000, 0].max(), 100)
        y_boundary = -(model.coef_[0][0] * x_boundary + model.intercept_) / model.coef_[0][1]
        if verbose:
            plt.plot(x_boundary, y_boundary, "r--")
            plt.xlabel("approximate dist")
            plt.ylabel("threshold dist")
            plt.title("binary classifier")
            plt.savefig(f'./figure/{dataset}/{dataset}_{method_type}_linear_{index_type}_{model_id}.png', dpi=500)

            plt.show()

        print("report：")
        print(classification_report(y_test, y_pred))

        w1 = model.coef_[0][0]
        w2 = model.coef_[0][1]
        b = model.intercept_[0]
        w1 /= -w2
        W_.append(w1)
        B_.append(b)

    f = open(save_path, 'w')
    print(len(W_), file=f)
    for i in range(len(W_)):
        print(str(W_[i]) + " " + str(B_[i]), file=f)

    f.close()
