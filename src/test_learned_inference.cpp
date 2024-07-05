#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>
#include "pca.h"
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
unsigned count_bound = 10000;
unsigned efSearch = 0;
int randomize = 0;
double recall = 0.999;
unsigned elements_bound = 5000000;

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
    char square_path[256] = "";
    char linear_path[256] = "";
    int subk = 0;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:k:e:p:i:q:g:r:t:p:v:s:n:l:", longopts, &ind);
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
            case 'v':
                if (optarg)strcpy(square_path, optarg);
                break;
            case 's':
                if (optarg) efSearch = atoi(optarg);
                break;
            case 'n':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'l':
                if (optarg)strcpy(linear_path, optarg);
                break;
        }
    }
    float *data;
    unsigned points_num, dim;
    load_float_data(dataset, data, points_num, dim);

    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    IVF ivf;
    ivf.load(index_path);
    Index_PCA::PCA PCA(ivf.N, ivf.d);
    PCA.base_dim = 32;
    PCA.dimension_ = Q.d;
    PCA.load_project_matrix(transformation_path);
    PCA.load_linear_model(linear_path);
    ivf.PCA = &PCA;
    if (randomize == 1) ivf.compute_base_square(true);
    PCA.project_vector(Q.data, G.n,true);
    count_bound = std::min(count_bound, (unsigned) G.n);
    std::cerr << "uni res :: " << PCA.uni_res_dim << endl;
    cerr << "test begin" << endl;

    std::vector<float> acc, thresh;
    std::vector<std::vector<float> > app;
    std::unordered_map<unsigned, bool> KNNmap;
    std::vector<unsigned> id_to_L1;

    unsigned sub_dim = 32;
    unsigned feature_dim = 2, model_count = Q.d / sub_dim;
    if (Q.d % sub_dim) model_count++;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << " sub dim:: " << sub_dim << endl;
    app.resize(model_count);
    id_to_L1.resize(ivf.N);
    for (int i = 0; i < ivf.N; i++) {
        id_to_L1[ivf.id[i]] = i;
    }
    unsigned base_count = subk * Q.n, bad_count = 0, test_bad_count = 0;
    cerr.setf(ios::fixed, ios::floatfield);
    for (int i = 0; i < G.n; i++) {
        float *q = Q.data + (long long)i * Q.d;
        if (randomize == 1) PCA.get_query_square(q);
        unsigned int *gt = G.data + (long long)i * G.d;
        float thresh_dist = naive_l2_dist_calc(q, ivf.res_data + (long long)id_to_L1[gt[subk - 1]] * Q.d, Q.d);
        thresh_dist += 1e-6;
        for (int j = 0; j < subk; j++) {
            unsigned L1_id = id_to_L1[gt[j]];
            float acc_dist = naive_l2_dist_calc(q, ivf.res_data + (long long)L1_id * Q.d, Q.d);
            float pre_sum=0, PCA_infer=0, app_dist=0;
            if (randomize == 1) {
                pre_sum = PCA.get_pre_sum(L1_id);
                PCA_infer = PCA.learned_fast_inference_lp(q, ivf.res_data + (long long)L1_id * Q.d, thresh_dist, 0, pre_sum);
            } else {
                PCA_infer = PCA.learned_fast_inference_l2(q, ivf.res_data + (long long)L1_id * Q.d, thresh_dist, 0, 0);
            }
            if (PCA_infer < 0) {
                test_bad_count++;
            }
            int tag_model = 0;
            if (randomize == 1) app_dist = pre_sum - 2 * naive_lp_dist_calc(q, ivf.res_data + (long long)L1_id * Q.d, 32);
            else app_dist = naive_l2_dist_calc(q, ivf.res_data + (long long)L1_id * Q.d, 32);
            if (PCA.learned_inference(app_dist, thresh_dist, tag_model)) {
                std::cerr << thresh_dist << " " << app_dist << " pca infer:: " << PCA_infer
                          << PCA.pre_query[32 - 1] << " cur dim:: "
                          << 32 << endl;
                bad_count++;
            }

            acc.push_back(acc_dist);
            thresh.push_back(thresh_dist);
            KNNmap[L1_id] = true;
        }
    }
    std::cerr<<test_bad_count<<" "<<bad_count<<std::endl;
    std::cerr << (float) bad_count / (float) base_count << endl;
    std::cerr << (float) test_bad_count / (float) base_count << endl;
    std::cerr << sub_dim << endl;

    return 0;
}
/*
 */