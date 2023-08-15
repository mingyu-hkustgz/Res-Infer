#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME
//#define USE_AVX
//#define USE_SSE

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

long double rotation_time=0;

void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const IVF &ivf, int k){
    float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
    struct rusage run_start, run_end;

    vector<int> nprobes;
    nprobes.push_back(25);
    nprobes.push_back(50);
    nprobes.push_back(75);
    nprobes.push_back(100);
    nprobes.push_back(125);
    nprobes.push_back(150);
    nprobes.push_back(175);
    nprobes.push_back(200);
    for(auto nprobe:nprobes){
        total_time=0;
        adsampling::clear();
        int correct = 0;

        for(int i=0;i<Q.n;i++){
            GetCurTime( &run_start);
            ResultHeap KNNs = ivf.search_with_pca(Q.data + i * Q.d, k, nprobe);
            GetCurTime( &run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
            // Recall
            while(KNNs.empty() == false){
                int id = KNNs.top().second;
                KNNs.pop();
                for(int j=0;j<k;j++)
                    if(id == G.data[i * G.d + j])correct ++;
            }
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);

        // (Search Parameter, Recall, Average Time/Query(us), Total Dimensionality)
        cout << nprobe << " " << recall * 100.00 << " " << time_us_per_query << " " << adsampling::tot_dimension << endl;
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
            // General Parameter
            {"help",                        no_argument,       0, 'h'},

            // Query Parameter
            {"randomized",                  required_argument, 0, 'd'},
            {"K",                           required_argument, 0, 'k'},
            {"epsilon0",                    required_argument, 0, 'e'},
            {"delta_d",                     required_argument, 0, 'p'},

            // Indexing Path
            {"dataset",                     required_argument, 0, 'n'},
            {"index_path",                  required_argument, 0, 'i'},
            {"query_path",                  required_argument, 0, 'q'},
            {"groundtruth_path",            required_argument, 0, 'g'},
            {"result_path",                 required_argument, 0, 'r'},
            {"transformation_path",         required_argument, 0, 't'},
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
    char linear_path[256] = "";

    int randomize = 0;
    int subk = 0;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:b:l:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'e':
                if(optarg)adsampling::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg)adsampling::delta_d = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
            case 'b':
                if (optarg)strcpy(codebook_path, optarg);
                break;
            case 'l':
                if (optarg)strcpy(linear_path, optarg);
                break;
        }
    }

    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);

    IVF ivf;
    ivf.load(index_path);
    Index_PCA::PCA PCA(ivf.N,ivf.D);
    PCA.load_project_matrix(transformation_path);

    StopW stopw = StopW();
    PCA.project_vector(Q.data, Q.n);
    rotation_time = stopw.getElapsedTimeMicro() / Q.n;
    Linear::Linear L(ivf.D);

    L.load_linear_model(linear_path);
    ivf.PCA = &PCA;
    ivf.PCA->proj_dim = 960;
    ivf.L = &L;
    std::cerr<<"rotate time:: "<<rotation_time<<std::endl;
    freopen(result_path,"a",stdout);
    test(Q, G, ivf, subk);
    return 0;
}