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
double recall = 0.9999;

vector<vector<tuple<unsigned, float, float> > > test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
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
                if (optarg)efSearch=atoi(optarg);
                break;
        }
    }


    Matrix<float> Q(query_path);
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
    std::vector<unsigned> dim_tag;
    std::vector<std::vector<unsigned> > cnt_tag;
    unsigned feature_dim = 3, model_count = 1;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count<<endl;
    unsigned all_items = 0;
    cnt_tag.resize(count_bound);
    for (int i = 0; i < count_bound; i++) {
        cnt_tag[i].resize(res[i].size());
        for (int j = 0; j < res[i].size(); j++) {
            cnt_tag[i][j] = all_items;
            all_items++;
        }
    }
    acc.resize(all_items);
    app.resize(all_items);
    cluster.resize(all_items);
    thresh.resize(all_items);
    for (int i = 0; i < count_bound; i++) {
        float *q = Q.data + i * Q.d;
        appr_alg->PQ->calc_dist_map(q);
        for (int j = 0; j < res[i].size(); j++) {
            auto u = res[i][j];
            unsigned tag = cnt_tag[i][j];
            unsigned id = get<0>(u);
            float node_dist = get<1>(u);
            float thresh_dist = get<2>(u);
            app[tag] = appr_alg->PQ->naive_product_map_dist(id);
            acc[tag] = node_dist;
            cluster[tag] = appr_alg->PQ->node_cluster_dist_[id];
            thresh[tag] = thresh_dist;
        }
    }
    std::ofstream out(logger_path, std::ios::binary);
    for (int i = 0; i < count_bound; i++) {
        for (int j = 0; j < res[i].size(); j++) {
            auto u = res[i][j];
            unsigned tag = cnt_tag[i][j];
            float node_dist = get<1>(u);
            float thresh_dist = get<2>(u);
            out.write((char *) &feature_dim, sizeof(unsigned));
            out.write((char *) &node_dist, sizeof(float));
            out.write((char *) &app[tag], sizeof(float));
            out.write((char *) &cluster[tag], sizeof(float));
            out.write((char *) &thresh_dist, sizeof(float));
        }
    }
    out.close();

    std::cerr << "save finished" << endl;
    if (isFileExists_ifstream((linear_path))) {
        Linear::Linear L(Q.d);
        L.recall = recall;
        L.load_linear_model(linear_path);
        std::ofstream fout(linear_path);
        fout << L.model_count << endl;
        L.binary_search_multi_linear(acc.size(), app.data(), acc.data(),cluster.data(), thresh.data());
        fout << L.W_[0]<<" "<<L.W_[1] << " " << L.B_[0] << endl;
    }
    return 0;
}
