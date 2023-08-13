/*
We highlight the function search which are closely related to our proposed algorithms.
We have included detailed comments in these functions. 

We explain the important variables for the enhanced IVF as follows.
1. d - It represents the number of initial dimensions.
    * For IVF  , d = D. 
    * For IVF+ , d = 0. 
    * For IVF++, d = delta_d. 
2. L1_data - The array to store the first d dimensions of a vector.
3. res_data - The array to store the remaining dimensions of a vector.

*/
#include <limits>
#include <queue>
#include <vector>
#include <algorithm>
#include <map>

#include "adsampling.h"
#include "matrix.h"
#include "utils.h"
#include "pq.h"
#include "pca.h"
#include "linear.h"
class IVF{
public:
    size_t N;
    size_t D;
    size_t C;
    size_t d; // the dimensionality of first a few dimensions

    float* L1_data;
    float* res_data;
    float* centroids;

    size_t* start;
    size_t* len;
    size_t* id;

    Index_PQ::Quantizer *PQ;
    Index_PCA::PCA *PCA;
    Linear::Linear *L;

    IVF();
    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive=0);
    ~IVF();

    ResultHeap search(float* query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    ResultHeap search_with_quantizer(float* query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    ResultHeap search_with_pca(float* query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    ResultHeap search_with_quantizer_simd(float* query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    std::vector<std::tuple<unsigned, float, float> > search_logger(float* query, size_t k, size_t nprobe) const;

    void encoder_origin_data();

    void save(char* filename);

    void load(char* filename);
};

IVF::IVF(){
    N = D = C = d = 0;
    start = len = id = NULL;
    L1_data = res_data = centroids = NULL;
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive){

    N = X.n;
    D = X.d;
    C = _centroids.n;

    assert(D > 32);
    start = new size_t [C];
    len   = new size_t [C];
    id    = new size_t [N];

    std::vector<size_t> * temp = new std::vector<size_t> [C];
    unsigned check_point = 0;
#pragma omp parallel for
    for(int i=0;i<X.n;i++){
        int belong = 0;
        float dist_min = X.dist(i, _centroids, 0);
        for(int j=1;j<C;j++){
            float dist = X.dist(i, _centroids, j);
            if(dist < dist_min){
                dist_min = dist;
                belong = j;
            }
        }
#pragma omp critical
        {
            check_point++;
            if(check_point % 50000 == 0){
                std::cerr << "Processing - " << check_point << " / " << X.n  << std::endl;
            }
            temp[belong].push_back(i);
        }
    }
    std::cerr << "Cluster Generated!" << std::endl;

    size_t sum = 0;
    for(int i=0;i<C;i++){
        len[i] = temp[i].size();
        start[i] = sum;
        sum += len[i];
        for(int j=0;j<len[i];j++){
            id[start[i] + j] = temp[i][j];
        }
    }

    if(adaptive == 1) d = 32;        // IVF++ - optimize cache (d = 32 by default)
    else if(adaptive == 0) d = D;   // IVF   - plain scan
    else d = 0;                     // IVF+  - plain ADSampling        

    L1_data   = new float [N * d + 1];
    res_data  = new float [N * (D - d) + 1];
    centroids = new float [C * D];
#pragma omp parallel for
    for(int i=0;i<N;i++){
        int x = id[i];
        for(int j=0;j<D;j++){
            if(j < d) L1_data[i * d + j] = X.data[x * D + j];
            else res_data[i * (D-d) + j - d] = X.data[x * D + j];
        }
    }

    std::memcpy(centroids, _centroids.data, C * D * sizeof(float));
    delete [] temp;

}

IVF::~IVF(){
    if(id != NULL)delete [] id;
    if(len != NULL)delete [] len;
    if(start != NULL)delete [] start;
    if(L1_data != NULL)delete [] L1_data;
    if(res_data != NULL)delete [] res_data;
    if(centroids != NULL)delete [] centroids;
}

ResultHeap IVF::search(float* query, size_t k, size_t nprobe, float distK) const{
    // the default value of distK is +inf 
    Result* centroid_dist = new Result [C];

    // Find out the closest N_{probe} centroids to the query vector.
    for(int i=0;i<C;i++){
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids+i*D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for(int i=0;i<nprobe;i++)
        ncan += len[centroid_dist[i].second];
    if(d == D)adsampling::tot_dimension += 1ll * ncan * D;
    float * dist = new float [ncan];
    Result * candidates = new Result [ncan];
    int * obj= new int [ncan];

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D. 
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32). 
    int cur = 0;
    for(int i=0;i<nprobe;i++){
        int cluster_id = centroid_dist[i].second;
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = sqr_dist(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if(d > 0)dist[cur] = tmp_dist;
            else dist[cur] = 0;
            obj[cur] = can;
            cur ++;
        }
    }
    ResultHeap KNNs;

    // d == D indicates FDScanning. 
    if(d == D){
        for(int i=0;i<ncan;i++){
            candidates[i].first = dist[i];
            candidates[i].second = id[obj[i]];
        }
        std::partial_sort(candidates, candidates + k, candidates + ncan);

        for(int i=0;i<k;i++){
            KNNs.emplace(candidates[i].first, candidates[i].second);
        }
    }
    // d < D indicates ADSampling with and without cache-level optimization
    if(d < D){
        auto cur_dist = dist;
        for(int i=0;i<nprobe;i++){
            int cluster_id = centroid_dist[i].second;
            for(int j=0;j<len[cluster_id];j++){
                size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                float tmp_dist = adsampling::dist_comp(distK, res_data + can * (D-d), query + d, *cur_dist, d);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                if(tmp_dist > 0){
                    KNNs.emplace(tmp_dist, id[can]);
                    if(KNNs.size() > k) KNNs.pop();
                }
                if(KNNs.size() == k && KNNs.top().first < distK){
                    distK = KNNs.top().first;
                }
                cur_dist++;
            }
        }
    }

    delete [] centroid_dist;
    delete [] dist;
    delete [] candidates;
    delete [] obj;
    return KNNs;
}


ResultHeap IVF::search_with_quantizer(float* query, size_t k, size_t nprobe, float distK) const{
    // the default value of distK is +inf
    Result* centroid_dist = new Result [C];

    // Find out the closest N_{probe} centroids to the query vector.
    for(int i=0;i<C;i++){
        centroid_dist[i].first = sqr_dist(query, centroids+i*D, D);
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    // fast inference by PQ approximate distance
    PQ->calc_dist_map(query);
    ResultHeap KNNs;
    float thresh = distK;
    for(int i=0;i<nprobe;i++){
        unsigned cluster_id = centroid_dist[i].second;
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
            if(L->linear_classifier_default_pq(PQ->naive_product_map_dist(can) - PQ->node_cluster_dist_[can], thresh)) continue;
            float tmp_dist = sqr_dist(query, L1_data + can * D, D);
            if(KNNs.size() < k) KNNs.emplace(tmp_dist,id[can]);
            else if(tmp_dist < KNNs.top().first){
                KNNs.emplace(tmp_dist,id[can]);
                if(KNNs.size() > k) KNNs.pop();
                thresh = KNNs.top().first;
            }
        }
    }

    delete [] centroid_dist;
    return KNNs;
}

ResultHeap IVF::search_with_pca(float* query, size_t k, size_t nprobe, float distK) const{
    // the default value of distK is +inf
    Result* centroid_dist = new Result [C];

    // Find out the closest N_{probe} centroids to the query vector.
    for(int i=0;i<C;i++){
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids+i*D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for(int i=0;i<nprobe;i++)
        ncan += len[centroid_dist[i].second];
    if(d == D)adsampling::tot_dimension += 1ll * ncan * D;
    float * dist = new float [ncan];
    Result * candidates = new Result [ncan];
    int * obj= new int [ncan];

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32).
    int cur = 0;
    for(int i=0;i<nprobe;i++){
        int cluster_id = centroid_dist[i].second;
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = sqr_dist(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if(d > 0)dist[cur] = tmp_dist;
            else dist[cur] = 0;
            obj[cur] = can;
            cur ++;
        }
    }
    ResultHeap KNNs;

    // d < D indicates ADSampling with and without cache-level optimization
    unsigned tag_model;
    if(d == 0) tag_model = 0;
    else tag_model = 1;
    if(d < D){
        auto cur_dist = dist;
        for(int i=0;i<nprobe;i++){
            int cluster_id = centroid_dist[i].second;
            for(int j=0;j<len[cluster_id];j++){
                size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                float tmp_dist = L->multi_linear_classifier_(query + d, res_data + can * (D-d), distK, tag_model, *cur_dist);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                if(tmp_dist > 0){
                    KNNs.emplace(tmp_dist, id[can]);
                    if(KNNs.size() > k) KNNs.pop();
                }
                if(KNNs.size() == k && KNNs.top().first < distK){
                    distK = KNNs.top().first;
                }
                cur_dist++;
            }
        }
    }

    delete [] centroid_dist;
    delete [] dist;
    delete [] candidates;
    delete [] obj;
    return KNNs;
}

//float tmp_dist = L->multi_linear_classifier_(query, res_data + can * D, thresh);
ResultHeap IVF::search_with_quantizer_simd(float* query, size_t k, size_t nprobe, float distK) const{
    // the default value of distK is +inf
    Result* centroid_dist = new Result [C];

    // Find out the closest N_{probe} centroids to the query vector.
    for(int i=0;i<C;i++){
        centroid_dist[i].first = sqr_dist(query, centroids+i*D, D);
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    // fast inference by PQ approximate distance
    PQ->calc_dist_map(query);
    ResultHeap KNNs;
    float thresh = distK;
    for(int i=0;i<nprobe;i++){
        unsigned cluster_id = centroid_dist[i].second;
#ifdef USE_AVX
        std::vector<unsigned> ids;
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
            ids.push_back(id[can]);
        }
        while(ids.size()%8!=0) ids.push_back(0);
        auto res = PQ->avx8_dist_sacn(ids.data(), ids.size());
#else
        std::vector<unsigned> ids;
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
            ids.push_back(can);
        }
        while(ids.size()%4!=0) ids.push_back(0);
        auto res = PQ->sse4_dist_sacn(ids.data(), ids.size());
//#else
//        auto res = new float[len[cluster_id]];
//        for(int j=0;j<len[cluster_id];j++){
//            size_t can = start[cluster_id] + j;
//            res[j] = PQ->naive_product_map_dist(id[can]);
//        }
#endif
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
            if(L->linear_classifier_default_pq(res[j] - PQ->node_cluster_dist_[can], thresh)) continue;
            float tmp_dist = sqr_dist(query, L1_data + can * D, D);
            if(KNNs.size() < k) KNNs.emplace(tmp_dist,id[can]);
            else if(tmp_dist < KNNs.top().first){
                KNNs.emplace(tmp_dist,id[can]);
                if(KNNs.size() > k) KNNs.pop();
                thresh = KNNs.top().first;
            }
        }
        delete [] res;
    }
    delete [] centroid_dist;
    return KNNs;
}


std::vector<std::tuple<unsigned, float, float> > IVF::search_logger(float* query, size_t k, size_t nprobe) const{
    // the default value of distK is +inf
    auto* centroid_dist = new Result [C];
    std::vector<std::tuple<unsigned ,float, float> > search_logger;
    // Find out the closest N_{probe} centroids to the query vector.
    for(int i=0;i<C;i++){
        centroid_dist[i].first = sqr_dist(query, centroids+i*D, D);
        centroid_dist[i].second = i;
    }
    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for(int i=0;i<nprobe;i++)
        ncan += len[centroid_dist[i].second];
    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    std::priority_queue<float> res_queue;
    for(int i=0;i<nprobe;i++){
        int cluster_id = centroid_dist[i].second;
        for(int j=0;j<len[cluster_id];j++){
            size_t can = start[cluster_id] + j;
            float tmp_dist;
            if(d==D)
                tmp_dist = sqr_dist(query, L1_data + can * d, d);
            else
                tmp_dist = sqr_dist(query, res_data + can * D, D);
            if(res_queue.size() < k) res_queue.push(tmp_dist);
            else if(tmp_dist < res_queue.top()){
                res_queue.push(tmp_dist);
                if(res_queue.size() > k) res_queue.pop();
            }
            if(res_queue.size()==k)
                search_logger.emplace_back(can, tmp_dist, res_queue.top());
        }
    }
    delete [] centroid_dist;
    return search_logger;
}



void IVF::save(char * filename){
    std::ofstream output(filename, std::ios::binary);

    output.write((char *) &N, sizeof(size_t));
    output.write((char *) &D, sizeof(size_t));
    output.write((char *) &C, sizeof(size_t));
    output.write((char *) &d, sizeof(size_t));

    if(d > 0)output.write((char *) L1_data,  N * d       * sizeof(float));
    if(d < D)output.write((char *) res_data, N * (D - d) * sizeof(float));
    output.write((char *) centroids, C * D * sizeof(float));

    output.write((char *) start, C * sizeof(size_t));
    output.write((char *) len  , C * sizeof(size_t));
    output.write((char *) id   , N * sizeof(size_t));

    output.close();
}

void IVF::load(char * filename){
    std::ifstream input(filename, std::ios::binary);
    cerr << filename << endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.read((char *) &N, sizeof(size_t));
    input.read((char *) &D, sizeof(size_t));
    input.read((char *) &C, sizeof(size_t));
    input.read((char *) &d, sizeof(size_t));
    cerr << N << " " << D << " " << C << " " << d << endl;

    L1_data   = new float [N * d + 10];
    res_data  = new float [N * (D - d) + 10];
    centroids = new float [C * D];

    start = new size_t [C];
    len   = new size_t [C];
    id    = new size_t [N];

    if(d > 0)input.read((char *) L1_data,  N * d       * sizeof(float));
    if(d < D)input.read((char *) res_data, N * (D - d) * sizeof(float));
    input.read((char *) centroids, C * D * sizeof(float));

    input.read((char *) start, C * sizeof(size_t));
    input.read((char *) len  , C * sizeof(size_t));
    input.read((char *) id   , N * sizeof(size_t));

    input.close();
}

void IVF::encoder_origin_data() {
    PQ->pq_mp = new unsigned char[PQ->nd_ * PQ->sub_vector];
    PQ->node_cluster_dist_ = new float[PQ->nd_];
    double ave_dist = 0.0;
#pragma omp parallel for
    for (int i = 0; i < PQ->nd_; i++) {
        float dist_to_centroid = 0.0;
        for (int j = 0; j < PQ->sub_vector; j++) {
            uint8_t belong = 0;
            float dist = naive_l2_dist_calc(L1_data + i * PQ->dimension_, PQ->pq_book[j][0].data(), PQ->sub_dim);
            for (int k = 1; k < PQ->sub_cluster_count; k++) {
                float new_dist = naive_l2_dist_calc(L1_data + i * D + j * PQ->sub_dim, PQ->pq_book[j][k].data(),
                                                    PQ->sub_dim);
                if (new_dist < dist) {
                    belong = k;
                    dist = new_dist;
                }
            }
            dist_to_centroid += dist;
            PQ->pq_mp[i * PQ->sub_vector + j] = belong;
        }
        PQ->node_cluster_dist_[i] = dist_to_centroid;
#pragma omp critical
        ave_dist += dist_to_centroid;
    }
    std::cerr << "Encoder ave dist:: " << ave_dist / PQ->nd_ << std::endl;
}
