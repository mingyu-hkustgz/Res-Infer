#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <hnswlib/hnswlib.h>
#include <adsampling.h>
#include "pq.h"
#include "pca.h"
#include "linear.h"
#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time = 0;
unsigned count_bound = 10000;
unsigned efSearch = 0;
double recall = 0.999;
unsigned elements_bound = 5000000;

vector<vector<tuple<unsigned, float, float> > >
test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive = 1) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();
    vector<vector<tuple<unsigned, float, float> > > hnsw_logger(qsize);
#pragma omp parallel for
    for (int i = 0; i < count_bound; i++) {
        std::vector<std::tuple<unsigned, float, float> > result = appr_alg.searchKnnlogger(massQ + vecdim * i, k,
                                                                                           adaptive);
        hnsw_logger[i] = result;
    }
    return hnsw_logger;
}

vector<vector<tuple<unsigned, float, float> > >
test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs;
    efs.push_back(efSearch);
    vector<vector<tuple<unsigned, float, float> > > hnsw_logger(qsize);
    for (size_t ef: efs) {
        appr_alg.setEf(ef);
        hnsw_logger = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
    }
    return hnsw_logger;
}

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

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";
    char codebook_path[256] = "";
    char linear_path[256] = "";
    char logger_path[256] = "";
    int randomize = 0;
    int subk = 100;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:b:l:o:s:", longopts, &ind);
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
            case 'p':
                if (optarg)adsampling::delta_d = atoi(optarg);
                break;
            case 'i':
                if (optarg)strcpy(index_path, optarg);
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
            case 'n':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'b':
                if (optarg)strcpy(codebook_path, optarg);
                break;
            case 'l':
                if (optarg)strcpy(linear_path, optarg);
                break;
            case 'o':
                if (optarg)strcpy(logger_path, optarg);
                break;
            case 's':
                if (optarg)efSearch = atoi(optarg);
                break;
        }
    }

    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);

    L2Space l2space(Q.d);
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);
    std::cerr << appr_alg->cur_element_count << " " << Q.d << std::endl;
    Index_PCA::PCA PCA(appr_alg->cur_element_count, Q.d);
    PCA.load_project_matrix(transformation_path);
    count_bound = std::min(count_bound, (unsigned) Q.n);
    PCA.project_vector(Q.data, count_bound);
    appr_alg->PCA = &PCA;

    cerr << "test begin" << endl;
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    vector<vector<tuple<unsigned, float, float> > > res(Q.n);
    res = test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);

    unsigned sub_dim = 32;
    unsigned feature_dim = 2, model_count = Q.d / sub_dim;
    if (Q.d % sub_dim) model_count++;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << " sub dim:: " << sub_dim << endl;
    std::cerr << "target recall:: " << recall << endl;
    std::vector<float> acc, thresh;
    std::vector<std::vector<float> > app;
    std::vector<unsigned> id_to_L1;

    app.resize(model_count);
    id_to_L1.resize(appr_alg->cur_element_count);
    for (int i = 0; i < id_to_L1.size(); i++) {
        id_to_L1[appr_alg->getExternalLabel(i)] = i;
    }
    std::ofstream fout(result_path, ios::app);
    fout.setf(ios::fixed, ios::floatfield);
    fout.precision(6);
    if (isFileExists_ifstream((linear_path))) {
        Linear::Linear L(Q.d);
        L.load_linear_model(linear_path);
        std::cerr << "finished load linear model " << endl;
        vector<double> base_prune(model_count), count_prune(model_count), ave_error(model_count), varr(model_count);
        for (int i = 0; i < count_bound; i++) {
            if (i % (count_bound / 20) == 0) std::cerr << "current analysis :: " << i << endl;
            float *q = Q.data + i * Q.d;
            unsigned int *gt = G.data + i * G.d;
            float thresh_dist = naive_l2_dist_calc(q, (float *) appr_alg->getDataByInternalId(id_to_L1[gt[subk - 1]]),
                                                   Q.d);
            for (auto u: res[i]) {
                unsigned L1_id = get<0>(u);
                float node_dist = get<1>(u);
                float app_dist = 0;
                unsigned app_count = 0;
                auto *p = (float *) appr_alg->getDataByInternalId(L1_id);
                for (unsigned k = 0; k < Q.d; k += sub_dim) {
                    if (k + sub_dim > Q.d) app_dist += naive_l2_dist_calc(q + k, p + k, Q.d % sub_dim);
                    else app_dist += naive_l2_dist_calc(q + k, p + k, sub_dim);
                    base_prune[app_count] += 1.0;
                    float reconstruct = app_dist * L.W_[app_count] + L.B_[app_count];
                    if (reconstruct > thresh_dist) count_prune[app_count] += 1.0;
                    ave_error[app_count] += abs(node_dist - reconstruct);
                    varr[app_count] += (node_dist - reconstruct) * (node_dist - reconstruct);
                    app_count++;
                }
            }
        }
        for (int i = 0; i < model_count; i++) {
            double rate_prune = count_prune[i] / base_prune[i];
            ave_error[i] /= base_prune[i];
            varr[i] /= base_prune[i];
            fout << "pca dim:: " << (i + 1) * sub_dim << " base count:: " << base_prune[i] << endl;
            fout << "prune rate:: " << rate_prune << endl;
            fout << "ave error:: " << ave_error[i] << endl;
            fout << "error variance:: " << varr[i] << endl;
        }
    }
    return 0;
}
