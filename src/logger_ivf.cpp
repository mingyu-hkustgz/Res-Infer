#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>
#include "pq.h"
#include "pca.h"
#include "linearmodel.h"
#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <ivf/ivf.h>
#include <adsampling.h>
#include <getopt.h>

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;

vector<vector<tuple<unsigned, float, float> > >  test_logger(const Matrix<float> &Q, const IVF &ivf, int k) {
    vector<int> nprobes;
    nprobes.push_back(100);
    vector<vector<tuple<unsigned, float, float> > > ivf_logger(Q.n);
    for (auto nprobe: nprobes) {
#pragma omp parallel for
        for (int i = 0; i < Q.n; i++) {
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
            {"codebook_path", required_argument, 0, 'b'},
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

    int randomize = 0;
    int subk = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:b:", longopts, &ind);
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
        }
    }
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);

    IVF ivf;
    ivf.load(index_path);
    Index_PCA::PCA PCA(ivf.N,ivf.D, ivf.res_data);
    PCA.load_project_matrix(transformation_path);
    PCA.project_vector(Q.data, Q.n);
    ivf.PCA = &PCA;
    auto res = test_logger(Q, ivf, subk);
    std::ofstream out("./logger/gist_logger_PCA_64_ivf.fvecs");
    unsigned feature_dim = 3;
    unsigned project_dim = 64;
    for(int i = 0; i < Q.n; i++){
        for(auto u:res[i]){
            unsigned id = get<0>(u);
            float node_dist = get<1>(u);
            float thresh_dist = get<2>(u);
            float app_dist = naive_l2_dist_calc(Q.data + i * ivf.D, ivf.res_data + id * ivf.D, project_dim);
            out.write((char*)&feature_dim,sizeof(unsigned));
            out.write((char*)&node_dist,sizeof(float));
            out.write((char*)&app_dist,sizeof(float));
            out.write((char*)&thresh_dist,sizeof(float));
        }
    }
    out.close();
//    float *data_load;
//    unsigned points_num, dim;
//    load_float_data(dataset, data_load, points_num, dim);
//    Index_PQ::Quantizer PQ(points_num, dim, data_load);
//    PQ.load_product_codebook(codebook_path);
//    PQ.encoder_origin_data();
//    PQ.load_project_matrix(transformation_path);
//    PQ.project_vector(Q.data, Q.n);
//    double count_all = 0.0, base_all = 0.0;
//    std::ofstream out("./logger/gist_logger_OPQ_120_ivf.fvecs");
//    unsigned feature_dim = 4;
//    for(int i = 0; i < Q.n; i++){
//        PQ.calc_dist_map(Q.data + i * dim);
//        for(auto u:res[i]){
//            unsigned id = get<0>(u);
//            float node_dist = get<1>(u);
//            float thresh_dist = get<2>(u);
//            float pq_dist = PQ.naive_product_map_dist(id);
//            float node_to_cluster = PQ.node_cluster_dist_[id];
//            out.write((char*)&feature_dim,sizeof(unsigned));
//            out.write((char*)&node_dist,sizeof(float));
//            out.write((char*)&pq_dist,sizeof(float));
//            out.write((char*)&node_to_cluster,sizeof(float));
//            out.write((char*)&thresh_dist,sizeof(float));
//            if(thresh_dist < pq_dist - node_to_cluster) count_all += 1.0;
//            base_all += 1.0;
//        }
//    }
//    std::cout<<count_all<<" "<<base_all<<endl;
//    std::cout<<count_all/base_all<<endl;
//    out.close();
    return 0;
}