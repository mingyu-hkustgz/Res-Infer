#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
#define COUNT_PRUNE_RATE
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
#include <ivf/ivf.h>
#include <adsampling.h>
#include <getopt.h>

using namespace std;

const int MAXK = 100;
int randomize = 0;
long double rotation_time = 0;
int efSearch = 0;

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k) {
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;

    vector<int> nprobes;
    unsigned efBase = efSearch;
    for (int i = 0; i < 8; i++) {
        nprobes.push_back(efBase);
        efBase += efSearch;
    }
    for (auto nprobe: nprobes) {
        total_time = 0;
        adsampling::clear();
        int correct = 0;

        for (int i = 0; i < Q.n; i++) {
            ResultHeap KNNs;
            GetCurTime(&run_start);
            if (randomize <= 2)
                KNNs = ivf.search(Q.data + i * Q.d, k, nprobe);
            else if (randomize == 3)
                KNNs = ivf.search_with_quantizer(Q.data + i * Q.d, k, nprobe);
            else if (randomize == 4)
                KNNs = ivf.search_with_quantizer_simd(Q.data + i * Q.d, k, nprobe);
            else
                KNNs = ivf.search_with_pca(Q.data + i * Q.d, k, nprobe);
            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
            // Recall
            while (KNNs.empty() == false) {
                int id = KNNs.top().second;
                KNNs.pop();
                for (int j = 0; j < k; j++)
                    if (id == G.data[i * G.d + j])correct++;
            }
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);

        // (Search Parameter, Recall, Average Time/Query(us), Total Dimensionality)
        if(randomize<=2|| randomize==5)
            cout << nprobe << " " << recall * 100.00 << " " << time_us_per_query << " " << adsampling::tot_dimension << endl;
        else
            cout << nprobe << " " << recall * 100.00 << " " << time_us_per_query << " " << (double)adsampling::tot_dist_calculation / (double)adsampling::tot_pq_dist << endl;
        if(recall * 100.00 > 99.5) break;
    }
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
    int subk = 0;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p::b:l:s:", longopts, &ind);
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
            case 's':
                if(optarg) efSearch = atoi(optarg);
                break;
        }
    }
    
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    IVF ivf;
    ivf.load(index_path);
    if(1<=randomize&&randomize<=2){
        Matrix<float> P(transformation_path);
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        adsampling::D = Q.d;
    }else if(3<=randomize&&randomize<=4){
        auto PQ = new Index_PQ::Quantizer(ivf.N,Q.d);
        PQ->load_product_codebook(codebook_path);
        PQ->load_project_matrix(transformation_path);
        auto L = new Linear::Linear(Q.d);
        L->load_linear_model(linear_path);
        ivf.L = L;
        ivf.PQ = PQ;
        ivf.encoder_origin_data();
        StopW stopw = StopW();
        PQ->project_vector(Q.data, Q.n);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        std::cerr << "rotate time:: " << rotation_time << std::endl;
        double dist_map_time;
        StopW stopw0 = StopW();
        for (int i = 0; i < Q.n; i++)
            PQ->calc_dist_map(Q.data + i * Q.d);
        dist_map_time = stopw0.getElapsedTimeMicro() / Q.n;;
        std::cerr << "dist map time::" << dist_map_time << std::endl;
    } else if (5 <= randomize && randomize <= 6) {
        auto PCA = new Index_PCA::PCA(ivf.N, Q.d);
        PCA->load_project_matrix(transformation_path);
        StopW stopw = StopW();
        PCA->project_vector(Q.data, Q.n);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        auto L = new Linear::Linear(Q.d);
        L->load_linear_model(linear_path);
        ivf.PCA = PCA;
        ivf.L = L;
        std::cerr<<"rotate time:: "<<rotation_time<<std::endl;
    }
    freopen(result_path,"a",stdout);
    test(Q, G, ivf, subk);
    return 0;
}