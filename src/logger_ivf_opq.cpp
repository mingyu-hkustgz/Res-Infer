#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>
#include "pq.h"
#include "pca.h"
#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <ivf/ivf.h>
#include <adsampling.h>
#include <getopt.h>
#include "linear.h"

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;
unsigned count_bound = 1000;

vector<vector<tuple<unsigned, float, float> > > test_logger(const Matrix<float> &Q, const IVF &ivf, int k) {
    vector<int> nprobes;
    nprobes.push_back(100);
    vector<vector<tuple<unsigned, float, float> > > ivf_logger(count_bound);
    for (auto nprobe: nprobes) {
#pragma omp parallel for
        for (int i = 0; i < count_bound; i++) {
            std::vector<std::tuple<unsigned, float, float> > cur;
            cur = ivf.search_logger(Q.data + i * Q.d, k, nprobe);
            ivf_logger[i] = cur;
        }
    }
    return ivf_logger;
}

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",                no_argument,       0, 'h'},

            // Query Parameter
            {"randomized",          required_argument, 0, 'd'},
            {"K",                   required_argument, 0, 'k'},
            {"epsilon0",            required_argument, 0, 'e'},
            {"delta_d",             required_argument, 0, 'p'},

            // Indexing Path
            {"dataset",             required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"query_path",          required_argument, 0, 'q'},
            {"groundtruth_path",    required_argument, 0, 'g'},
            {"result_path",         required_argument, 0, 'r'},
            {"transformation_path", required_argument, 0, 't'},
            {"codebook_path",       required_argument, 0, 'b'},
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
    int subk = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:b:l:o:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 'e':
                if (optarg)adsampling::epsilon0 = atof(optarg);
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
        }
    }
    Matrix<float> Q(query_path);
    IVF ivf;
    ivf.load(index_path);
    Index_PQ::Quantizer PQ(ivf.N, ivf.D);
    PQ.load_product_codebook(codebook_path);
    PQ.load_project_matrix(transformation_path);
    PQ.project_vector(Q.data, count_bound);
    ivf.PQ = &PQ;
    ivf.encoder_origin_data();
    cerr << "test begin" << endl;
    auto res = test_logger(Q, ivf, subk);
    std::vector<float> acc, thresh;
    std::vector<std::vector<float> > app;
    std::vector<unsigned> dim_tag;
    std::vector<std::vector<unsigned> > cnt_tag;
    unsigned sub_dim = 64;
    unsigned feature_dim = 2, model_count = 1;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << " sub dim:: " << sub_dim << endl;
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
    thresh.resize(all_items);
    app.resize(model_count);
    for (int i = 0; i < model_count; i++) app[i].resize(all_items);
    std::cerr << sub_dim << endl;
    for (int i = 0; i < count_bound; i++) {
        float *q = Q.data + i * Q.d;
        ivf.PQ->calc_dist_map(q);
        for (int j = 0; j < res[i].size(); j++) {
            auto u = res[i][j];
            unsigned tag = cnt_tag[i][j];
            unsigned id = get<0>(u);
            float node_dist = get<1>(u);
            float thresh_dist = get<2>(u);
            float app_dist = ivf.PQ->naive_product_map_dist(id) - ivf.PQ->node_cluster_dist_[id];
            app[0][tag] = app_dist;
            acc[tag] = node_dist;
            thresh[tag] = thresh_dist;
        }
    }
    std::ofstream out(logger_path,std::ios::binary);
    for (int i = 0; i < count_bound; i++) {
        for (int j = 0; j < res[i].size(); j++) {
            auto u = res[i][j];
            unsigned tag = cnt_tag[i][j];
            float node_dist = get<1>(u);
            float thresh_dist = get<2>(u);
            out.write((char *) &feature_dim, sizeof(unsigned));
            out.write((char *) &node_dist, sizeof(float));
            for (int k = 0; k < model_count; k++) {
                out.write((char *) &app[k][tag], sizeof(float));
            }
            out.write((char *) &thresh_dist, sizeof(float));
        }
    }
    out.close();

    std::cerr << "save finished" << endl;
    if (isFileExists_ifstream((linear_path))) {
        Linear::Linear L(ivf.D);
        L.recall = 0.9999;
        L.load_linear_model(linear_path);
        std::ofstream fout(linear_path);
        fout << L.model_count << endl;
        for (int i = 0; i < L.model_count; i++) {
            L.binary_search_single_linear(acc.size(), app[i].data(), acc.data(), thresh.data(), i);
            fout << L.W_[i] << " " << L.B_[i] << endl;
        }
    }
    return 0;
}