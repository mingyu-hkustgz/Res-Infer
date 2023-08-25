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
unsigned count_bound = 1000;
unsigned efSearch = 0;
double recall = 0.999;
unsigned elements_bound = 5000000;

vector<vector<tuple<unsigned, float, float> > >
test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
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
    Index_PQ::Quantizer PQ(appr_alg->cur_element_count, Q.d);
    PQ.load_product_codebook(codebook_path);
    PQ.load_project_matrix(transformation_path);
    count_bound = std::min(count_bound, (unsigned) Q.n);
    PQ.project_vector(Q.data, count_bound);
    appr_alg->PQ = &PQ;
    appr_alg->encoder_origin_data();

    cerr << "test begin" << endl;
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    vector<vector<tuple<unsigned, float, float> > > res(Q.n);
    res = test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);

    std::vector<float> acc, app, thresh, cluster;
    std::unordered_map<unsigned, bool> KNNmap;
    std::vector<unsigned> id_to_L1;
    unsigned feature_dim = 3, model_count = 1;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << endl;
    id_to_L1.resize(appr_alg->cur_element_count);
    for (int i = 0; i < id_to_L1.size(); i++) {
        id_to_L1[appr_alg->getExternalLabel(i)] = i;
    }
    for (int i = 0; i < count_bound; i++) {
        float *q = Q.data + i * Q.d;
        unsigned int *gt = G.data + i * G.d;
        appr_alg->PQ->calc_dist_map(q);
        float thresh_dist = naive_l2_dist_calc(q, (float *) appr_alg->getDataByInternalId(id_to_L1[gt[subk - 1]]), Q.d);
        for (int j = 0; j < subk; j++) {
            unsigned L1_id = id_to_L1[gt[j]];
            float acc_dist = naive_l2_dist_calc(q, (float *) appr_alg->getDataByInternalId(L1_id), Q.d);
            float app_dist = appr_alg->PQ->naive_product_map_dist(L1_id);
            float cluster_dist = appr_alg->PQ->node_cluster_dist_[L1_id];
            app.push_back(app_dist);
            acc.push_back(acc_dist);
            cluster.push_back(cluster_dist);
            thresh.push_back(thresh_dist);
            KNNmap[L1_id] = true;
        }
    }

    for (int i = 0; i < count_bound; i++) {
        float *q = Q.data + i * Q.d;
        unsigned int *gt = G.data + i * G.d;
        appr_alg->PQ->calc_dist_map(q);
        float thresh_dist = naive_l2_dist_calc(q, (float *) appr_alg->getDataByInternalId(id_to_L1[gt[subk - 1]]), Q.d);
        for (auto u: res[i]) {
            unsigned L1_id = get<0>(u);
            if (KNNmap[L1_id]) continue;
            float acc_dist = get<1>(u);
            float app_dist = appr_alg->PQ->naive_product_map_dist(L1_id);
            float cluster_dist = appr_alg->PQ->node_cluster_dist_[L1_id];
            app.push_back(app_dist);
            acc.push_back(acc_dist);
            cluster.push_back(cluster_dist);
            thresh.push_back(thresh_dist);
        }
        if (acc.size() > elements_bound) break;
    }


    std::ofstream out(logger_path, std::ios::binary);
    int knn_count = 0;
    for (int tag = 0; tag < acc.size(); tag++) {
        if (acc[tag] < (double) thresh[tag] + 1e-8) {
            out.write((char *) &feature_dim, sizeof(unsigned));
            out.write((char *) &acc[tag], sizeof(float));
            out.write((char *) &app[tag], sizeof(float));
            out.write((char *) &cluster[tag], sizeof(float));
            out.write((char *) &thresh[tag], sizeof(float));
            knn_count++;
        }
    }
    std::cerr << knn_count << endl;
    static std::default_random_engine Engine;
    static std::uniform_int_distribution<unsigned> rand(0, acc.size());
    for (int tag = 0; tag < std::min((unsigned long) elements_bound, acc.size()); tag++) {
        unsigned rand_index = rand(Engine);
        if (acc[rand_index] > thresh[rand_index]) {
            out.write((char *) &feature_dim, sizeof(unsigned));
            out.write((char *) &acc[rand_index], sizeof(float));
            out.write((char *) &app[rand_index], sizeof(float));
            out.write((char *) &cluster[rand_index], sizeof(float));
            out.write((char *) &thresh[rand_index], sizeof(float));
        }
    }
    out.close();

    std::cerr << "save finished" << endl;
    if (isFileExists_ifstream((linear_path))) {
        Linear::Linear L(Q.d);
        L.count_base = count_bound * subk;
        L.recall = recall;
        L.load_linear_model(linear_path);
        std::ofstream fout(linear_path);
        fout.setf(ios::fixed, ios::floatfield);
        fout.precision(6);
        fout << L.model_count << endl;
        L.binary_search_multi_linear(acc.size(), app.data(), acc.data(), cluster.data(), thresh.data());
        fout << L.W_[0] << " " << L.W_[1] << " " << L.B_[0] << endl;
    }
    return 0;
}
