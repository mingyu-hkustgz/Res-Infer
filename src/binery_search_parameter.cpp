#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <adsampling.h>
#include "pca.h"
#include <getopt.h>
#include <Eigen/Dense>

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;
unsigned count_bound = 10000;
double recall = 0.995;

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",                no_argument,       0, 'h'},

            // Query Parameter
            {"randomized",          required_argument, 0, 'd'},
            {"k",                   required_argument, 0, 'k'},
            {"epsilon0",            required_argument, 0, 'e'},
            {"gap",                 required_argument, 0, 'p'},

            // Indexing Path
            {"dataset",             required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"query_path",          required_argument, 0, 'q'},
            {"groundtruth_path",    required_argument, 0, 'g'},
            {"result_path",         required_argument, 0, 'r'},
            {"transformation_path", required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char data_path[256] = "";
    char transformation_path[256] = "";
    char linear_path[256] = "";
    int randomize = 1;//default as ip based
    int subk = 100;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:k:e:n:q:g:r:t:l:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 'e':
                if (optarg)recall = atof(optarg);
                break;
            case 'n':
                if (optarg)strcpy(data_path, optarg);
                break;
            case 'q':
                if (optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if (optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if (optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if (optarg)strcpy(transformation_path, optarg);
                break;
            case 'l':
                if (optarg)strcpy(linear_path, optarg);
                break;
        }
    }
    Matrix<float> N(data_path);
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    std::vector<float> acc, thresh;
    std::vector<std::vector<float> > app;

    auto PCA = new Index_PCA::PCA(N.n, N.d);
    PCA->base_dim = 32;
    PCA->load_project_matrix(transformation_path);

    if (randomize == 1) {
        PCA->base_square = new float[N.n];
        float *tmp_base = N.data;
        for (int i = 0; i < N.n; i++) {
            float res = 0.0;
            for (int j = 0; j < N.d; j++) {
                res += tmp_base[j] * tmp_base[j];
            }
            PCA->base_square[i] = res;
            tmp_base += N.d;
        }
    }
    PCA->project_vector(Q.data, G.n, true);
    unsigned sub_dim = 32;
    unsigned model_count = Q.d / sub_dim;
    PCA->model_count = model_count;
    if (Q.d % sub_dim) model_count++;
    app.resize(model_count);
    std::cerr << "test begin" << std::endl;
    count_bound = std::min(count_bound, (unsigned) G.n);
    for (long long i = 0; i < count_bound; i++) {
        float *q = Q.data + (long long) i * Q.d;
        unsigned int *gt = G.data + (long long) i * G.d;
        float *p = N.data + (long long) gt[subk - 1] * Q.d;
        float thresh_dist = naive_l2_dist_calc(p, q, Q.d);
        if (randomize == 1)
            PCA->get_query_square(q);
        for (int j = 0; j < subk; j++) {
            p = N.data + (long long) gt[j] * Q.d;
            float app_dist;
            float acc_dist = naive_l2_dist_calc(p, q, Q.d);
            if (randomize == 1) app_dist = PCA->get_pre_sum(gt[j]);
            else app_dist = 0;
            unsigned app_count = 0;
            for (unsigned k = 0; k < Q.d; k += sub_dim) {
                if (randomize == 1) {
                    if (k + sub_dim > Q.d) app_dist -= 2.0f * naive_lp_dist_calc(q + k, p + k, Q.d % sub_dim);
                    else app_dist -= 2.0f * naive_lp_dist_calc(q + k, p + k, sub_dim);
                } else {
                    if (k + sub_dim > Q.d) app_dist += naive_l2_dist_calc(q + k, p + k, Q.d % sub_dim);
                    else app_dist += naive_l2_dist_calc(q + k, p + k, sub_dim);
                }
                app[app_count].push_back(app_dist);
                app_count++;
            }
            acc.push_back(acc_dist);
            thresh.push_back(thresh_dist);
        }
    }
    PCA->W_.resize(model_count);
    PCA->B_.resize(model_count);
    PCA->b_.resize(model_count);
    Eigen::VectorXf Y = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(acc.data(), acc.size());
    float y_mean = Y.mean();
    for (int i = 0; i < model_count; i++) {
        if(randomize==0)
        {
            Eigen::VectorXf X = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(app[i].data(), app[i].size());
            float x_mean = X.mean();
            // compute OLS
            Eigen::VectorXf X_centered = X.array() - x_mean;
            Eigen::VectorXf Y_centered = Y.array() - y_mean;
            float w = (X_centered.cwiseProduct(Y_centered).sum()) / (X_centered.cwiseProduct(X_centered).sum());
            float b = y_mean - w * x_mean;
            std::cerr << "OLS w: " << w << std::endl;
            std::cerr << "OLS b: " << b << std::endl;
            PCA->W_[i] = w;
            PCA->B_[i] = b;
            PCA->b_[i] = b;
        }else{
            PCA->W_[i] = 1.0;
            PCA->B_[i] = 0;
            PCA->b_[i] = 0;
        }

    }
    std::cerr << " models:: " << model_count << " sub dim:: " << sub_dim << endl;
    std::cerr << "target recall:: " << recall << endl;
    double exp_recall = 1.0 - (1.0 - recall) / (model_count - 1.0);
    std::cerr << "save finished with recall:: " << recall << " " << exp_recall << endl;
    unsigned count_base = count_bound * subk;
    std::ofstream fout(linear_path);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(6);
    fout << PCA->model_count << endl;
    for (int i = 0; i < model_count; i++) {
        std::cerr << exp_recall << endl;
        PCA->recall = exp_recall;
        PCA->count_base = count_base;
        if (i == PCA->model_count - 1) {
            PCA->W_[model_count - 1] = 1.0;
            PCA->B_[model_count - 1] = 0;
            PCA->b_[model_count - 1] = 0;
        } else {
            PCA->binary_search_single_linear(acc.size(), app[i].data(), acc.data(), thresh.data(), i);
        }
        fout << PCA->W_[i] << " " << PCA->B_[i] << " " << PCA->b_[i] << endl;
    }
    return 0;
}
