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
    char logger_path[256] = "";
    int randomize = 1;//default as HNSW++
    int subk = 100;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:k:e:i:q:g:o:s:", longopts, &ind);
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
            case 'i':
                if (optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if (optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if (optarg)strcpy(groundtruth_path, optarg);
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

    count_bound = std::min(count_bound, (unsigned)Q.n);

    L2Space l2space(Q.d);
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);
    std::cerr << appr_alg->cur_element_count << " " << Q.d << std::endl;

    cerr << "test begin" << endl;
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    vector<vector<tuple<unsigned, float, float> > > res(Q.n);
    res = test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);
    std::unordered_map<unsigned, bool> KNNmap;
    std::vector<unsigned> id_to_L1;
    unsigned feature_dim = 2;

    id_to_L1.resize(appr_alg->cur_element_count);
    for (int i = 0; i < id_to_L1.size(); i++) {
        id_to_L1[appr_alg->getExternalLabel(i)] = i;
    }
    for (int i = 0; i < count_bound; i++) {
        unsigned int *gt = G.data + i * G.d;
        for (int j = 0; j < subk; j++) {
            unsigned L1_id = id_to_L1[gt[j]];
            KNNmap[L1_id] = true;
        }
    }
    std::vector<std::pair<unsigned, unsigned> > negative_sample;
    for (int i = 0; i < count_bound; i++) {
        std::random_shuffle(res[i].begin(), res[i].end());
        for (auto u: res[i]) {
            unsigned L1_id = get<0>(u);
            if (KNNmap[L1_id]) continue;
            negative_sample.emplace_back(i, appr_alg->getExternalLabel(L1_id));
        }
        if (negative_sample.size() >= elements_bound) break;
    }
    std::ofstream out(logger_path, std::ios::binary);
    for (int i = 0; i < elements_bound; i++) {
        out.write((char *) &feature_dim, sizeof(unsigned));
        out.write((char*)&negative_sample[i].first, sizeof(unsigned));
        out.write((char*)&negative_sample[i].second, sizeof(unsigned));
    }
    return 0;
}