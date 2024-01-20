import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from tqdm import tqdm

source = '/home/BLD/mingyu/DATA/vector_data'
datasets = ['_sift10m', 'gist', 'deep1M', '_tiny5m', '_glove2.2m', '_word2vec']
method = 'naive'
marker = ['o', 'triangle', 'square', 'otimes', 'star', 'diamond', 'pentagon', 'oplus', 'asterisk']


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
    out_put_file = open(f"result_tex_{method}.txt", "w")
    count = 0
    for dataset in datasets:
        print(f"visual - {dataset}")
        real_data = "NONE"
        base_log = 100
        if dataset == "gist":
            real_data = "GIST"
            base_log = 100
        if dataset == "deep1M":
            real_data = "DEEP"
            base_log = 100
        if dataset == "_sift10m":
            real_data = "SIFT"
            base_log = 100
        if dataset == "_glove2.2m":
            real_data = "GLOVE"
            base_log = 100
        if dataset == "_word2vec":
            real_data = "WORD2VEC"
            base_log = 100
        if dataset == "_tiny5m":
            real_data = "TINY"
            base_log = 100

        for K in [20, 100]:
            plt.figure(figsize=(12, 8))
            file_path = f"./results/recall@{K}/{dataset}"
            if not os.path.exists(file_path):
                continue
            print("% " + dataset + "@" + str(K), file=out_put_file)
            print('\\subfloat[' + real_data + '-HNSW]{', file=out_put_file)
            print(r'''
\begin{tikzpicture}[scale=1]
\begin{axis}[
    height=\columnwidth/2.50,
width=\columnwidth/1.90,
xlabel=recall@''' + str(K) + r''',
ylabel=Qpsx''' + str(base_log) + r''',
label style={font=\scriptsize},
tick label style={font=\scriptsize},
ymajorgrids=true,
xmajorgrids=true,
grid style=dashed,
]''', file=out_put_file)
            for i in range(9):
                result_path = f"./results/recall@{K}/{dataset}/{dataset}_ad_hnsw_{i}.log"
                if not os.path.exists(result_path):
                    continue
                recall, Qps = load_result_data(result_path)
                if i == 0:
                    print(
                        f"\\addplot[line width=0.15mm,color=amaranth,mark={marker[i]},mark size=0.5mm]%hnsw {dataset}",
                        file=out_put_file)
                elif i == 1:
                    print(f"\\addplot[line width=0.15mm,color=amber,mark={marker[i]},mark size=0.5mm]%hnsw++ {dataset}",
                          file=out_put_file)
                elif i == 2:
                    print(f"\\addplot[line width=0.15mm,color=black,mark={marker[i]},mark size=0.5mm]%hnsw+ {dataset}",
                          file=out_put_file)
                elif i == 3:
                    if method == "sse":
                        continue
                    print(
                        f"\\addplot[line width=0.15mm,color=forestgreen,mark={marker[i]},mark size=0.5mm]%hnsw-opq {dataset}",
                        file=out_put_file)
                elif i == 4:
                    if method == "naive":
                        continue
                    print(
                        f"\\addplot[line width=0.15mm,color=forestgreen,mark={marker[i]},mark size=0.5mm]%hnsw-opq-sse {dataset}",
                        file=out_put_file)
                elif i == 5:
                    print(
                        f"\\addplot[line width=0.15mm,color=aliceblue,mark={marker[i]},mark size=0.5mm]%hnsw-learn-pca {dataset}",
                        file=out_put_file)
                elif i == 6:
                    print(
                        f"\\addplot[line width=0.15mm,color=airforceblue,mark={marker[i]},mark size=0.5mm]%hnsw-learn-pca {dataset}",
                        file=out_put_file)
                elif i == 7:
                    print(
                        f"\\addplot[line width=0.15mm,color=navy,mark={marker[i]},mark size=0.5mm]%hnsw-res-infer++ {dataset}",
                        file=out_put_file)
                elif i == 8:
                    print(
                        f"\\addplot[line width=0.15mm,color=violate,mark={marker[i]},mark size=0.5mm]%hnsw-res-infer {dataset}",
                        file=out_put_file)
                print("plot coordinates {", file=out_put_file)
                for j in range(len(recall)):
                    print("    ( " + str(round(recall[j] / 100, 3)) + ", " + str(round(Qps[j] / base_log, 3)) + " )",
                          file=out_put_file)
                print("};", file=out_put_file)

            print(r'''
\end{axis}
\end{tikzpicture}\hspace{2mm}''', file=out_put_file)
            print('}', file=out_put_file)
            count += 1
            if count % 4 == 0:
                print(r'\\', file=out_put_file)
                print(r'\vspace{1mm}', file=out_put_file)

            print('\\subfloat[' + real_data + '-IVF]{', file=out_put_file)
            print(r'''
\begin{tikzpicture}[scale=1]
\begin{axis}[
    height=\columnwidth/2.5,
width=\columnwidth/1.90,
xlabel=recall@''' + str(K) + r''',
ylabel=Qpsx''' + str(base_log) + r''',
ylabel style={yshift=-0.5mm},
label style={font=\scriptsize},
tick label style={font=\scriptsize},
ymajorgrids=true,
xmajorgrids=true,
grid style=dashed,
]''', file=out_put_file)
            for i in range(7):
                result_path = f"./results/recall@{K}/{dataset}/{dataset}_ad_ivf_{i}.log"
                if not os.path.exists(result_path):
                    continue
                recall, Qps = load_result_data(result_path)
                if i == 0:
                    print(f"\\addplot[line width=0.15mm,color=red,mark={marker[i]},mark size=0.5mm]%ivf {dataset}",
                          file=out_put_file)
                elif i == 1:
                    print(f"\\addplot[line width=0.15mm,color=orange,mark={marker[i]},mark size=0.5mm]%ivf++ {dataset}",
                          file=out_put_file)
                elif i == 3:
                    print(
                        f"\\addplot[line width=0.15mm,color=airforceblue,mark={marker[i]},mark size=0.5mm]%ivf-learn-pca++ {dataset}",
                        file=out_put_file)
                elif i == 4:
                    print(
                        f"\\addplot[line width=0.15mm,color=navy,mark={marker[i]},mark size=0.5mm]%ivf-learn-pca++ {dataset}",
                        file=out_put_file)
                elif i == 5:
                    print(
                        f"\\addplot[line width=0.15mm,color=violate,mark={marker[i]},mark size=0.5mm]%ivf-res++ {dataset}",
                        file=out_put_file)
                elif i == 6:
                    print(
                        f"\\addplot[line width=0.15mm,color=forestgreen,mark={marker[i]},mark size=0.5mm]%ivf-opq {dataset}",
                        file=out_put_file)
                print("plot coordinates {", file=out_put_file)
                for j in range(len(recall)):
                    print("    ( " + str(round(recall[j] / 100, 3)) + ", " + str(round(Qps[j] / base_log, 3)) + " )",
                          file=out_put_file)
                print("};", file=out_put_file)
            print(r'''
\end{axis}
\end{tikzpicture}\hspace{2mm}''', file=out_put_file)
            print('}', file=out_put_file)

            count += 1
            if count % 4 == 0:
                print(r'\\', file=out_put_file)
                print(r'\vspace{1mm}', file=out_put_file)
