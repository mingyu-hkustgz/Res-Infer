#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
//#define COUNT_DIMENSION
//#define COUNT_PRUNE_RATE
/***
 * The operation to define use SSE
 ***/
//#define USE_SSE
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <hnswlib/hnswlib.h>
#include <adsampling.h>

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time = 0;
int efSearch = 0;
float finger_ratio = 1.5;
int finger_lsh_dim = 64;
double outer_recall = 0;

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
                   size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                   size_t subk, HierarchicalNSW<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(
                    appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(massQA[k * i + j]),
                                          appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result,
           std::priority_queue<std::pair<float, labeltype >> &gt) {
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
                        vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);
#endif
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k, adaptive);
#ifndef WIN32
        GetCurTime(&run_end);
        GetTime(&run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        correct += tmp;
    }
    long double time_us_per_query = total_time / qsize + rotation_time;
    long double recall = 1.0f * correct / total;

    cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " "
         << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    outer_recall = recall * 100;
    return;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
                           vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 8; i++) {
        efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.setEf(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
//        if(outer_recall > 99.5) break;
    }
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
    char transformation_path[256] = "";
    char codebook_path[256] = "";
    char dataset[256]= "";
    int randomize = 0;
    char square_path[256] = "";
    char linear_path[256] = "";
    int subk = 100;
    int delta_d = 32;
    float epsilon0 = 8.0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:k:e:p:b:i:q:g:r:t:p:v:s:l:n:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 'e':
                if (optarg)epsilon0 = atof(optarg);
                break;
            case 'p':
                if (optarg) delta_d = atoi(optarg);
                break;
            case 'b':
                if (optarg)strcpy(codebook_path, optarg);
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
            case 'l':
                if (optarg)strcpy(linear_path, optarg);
                break;
            case 's':
                if (optarg) efSearch = atoi(optarg);
                break;
            case 'n':
                if (optarg)strcpy(dataset, optarg);
                break;
        }
    }

    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    L2Space l2space(Q.d);
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, index_path, false);

    if (1 <= randomize && randomize <= 2) {
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    } else if (3 <= randomize && randomize <= 4) {
        std::cerr << appr_alg->cur_element_count << " " << Q.d << std::endl;
        auto PQ = new Index_PQ::Quantizer(appr_alg->cur_element_count, Q.d);
        PQ->load_product_codebook(codebook_path);
        PQ->load_project_matrix(transformation_path);
        PQ->load_linear_model(linear_path);
        appr_alg->PQ = PQ;
        appr_alg->encoder_origin_data();
        StopW stopw = StopW();
        PQ->project_vector(Q.data, Q.n);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
        std::cerr << "rotate time:: " << rotation_time << endl;
    } else if (5 <= randomize && randomize <= 6) {
        std::cerr << appr_alg->cur_element_count << " " << Q.d << std::endl;
        auto PCA = new Index_PCA::PCA(appr_alg->cur_element_count, Q.d);
        PCA->load_linear_model(linear_path);
        PCA->base_dim = delta_d;
        PCA->load_project_matrix(transformation_path);
        appr_alg->PCA = PCA;
        appr_alg->PCA->dimension_ = Q.d;
        PCA->load_linear_model(linear_path);
        if (randomize == 5){
            appr_alg->compute_base_square(true);
            appr_alg->reorganized_data_aligned();
        }
        StopW stopw = StopW();
        PCA->project_vector(Q.data, Q.n, true);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        std::cerr << "rotate time:: " << rotation_time << endl;
    } else if (7 <= randomize && randomize <= 8) {
        std::cerr << appr_alg->cur_element_count << " " << Q.d << std::endl;
        auto PCA = new Index_PCA::PCA(appr_alg->cur_element_count, Q.d);
        PCA->sigma_count = epsilon0;
        PCA->base_dim = delta_d;
        PCA->load_project_matrix(transformation_path);
        appr_alg->PCA = PCA;
        appr_alg->compute_base_square(false);
        if (randomize == 7) appr_alg->reorganized_data_aligned();
        StopW stopw = StopW();
        PCA->project_vector(Q.data, Q.n, false);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        std::cerr << "rotate time:: " << rotation_time << endl;
    }
    else if(randomize == 9){
        string base_path_str = "./DATA";
        string data_str(dataset);
        string M_str = "16";
        string ef_str = "500";
        string finger_projection_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" + data_str + "M" + M_str + "ef" + ef_str + "_LSH.fvecs";
        string finger_b_dres_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" + data_str + "M" + M_str + "ef" + ef_str +"_b_dres.fvecs";
        string finger_sgn_dres_P_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" + data_str + "M" + M_str + "ef" + ef_str +"_sgn_dres_P.ivecs";
        string finger_c_2_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" +  data_str + "M" + M_str + "ef" + ef_str +"_c_2.fvecs";
        string finger_c_P_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" +  data_str + "M" + M_str + "ef" + ef_str +"_c_P.fvecs";
        string finger_start_idx_path_str = base_path_str + "/" + data_str + "/FINGER" + to_string(finger_lsh_dim) +"_" +  data_str + "M" + M_str + "ef" + ef_str +"_start_idx.ivecs";
        char finger_projection_path[256] = "";
        strcpy(finger_projection_path, finger_projection_path_str.c_str());
        char finger_b_dres_path[256] = "";
        strcpy(finger_b_dres_path, finger_b_dres_path_str.c_str());
        char finger_sgn_dres_P_path[256] = "";
        strcpy(finger_sgn_dres_P_path, finger_sgn_dres_P_path_str.c_str());
        char finger_c_2_path[256] = "";
        strcpy(finger_c_2_path, finger_c_2_path_str.c_str());
        char finger_c_P_path[256] = "";
        strcpy(finger_c_P_path, finger_c_P_path_str.c_str());
        char finger_start_idx_path[256] = "";
        strcpy(finger_start_idx_path, finger_start_idx_path_str.c_str());
        finger::P = Matrix<float>(finger_projection_path);
        finger::bs_dres = Matrix<float>(finger_b_dres_path);
        finger::c_2s = Matrix<float>(finger_c_2_path);
        finger::c_Ps = Matrix<float>(finger_c_P_path);
        finger::start_idxs = Matrix<int>(finger_start_idx_path);
        finger::ratio = finger_ratio;
        finger::D = Q.d;
        finger::lsh_dim = finger_lsh_dim;
        Matrix<int> sgn_d_res_Ps = Matrix<int>(finger_sgn_dres_P_path);
        unsigned int edge_num = sgn_d_res_Ps.n;
        finger::binary_sgn_d_res_Ps = vector<unsigned long long>();
        for(int i = 0; i < edge_num; i++){
            finger::binary_sgn_d_res_Ps.push_back(
                    finger::get_binary_sgn_from_array(sgn_d_res_Ps[i])
            );
        }
        finger::pre_compute_cos();
        StopW stopw = StopW();
        finger::q_Ps = mul(Q, finger::P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
    }
    freopen(result_path, "a", stdout);
    size_t k = G.d;
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);

    return 0;
}
