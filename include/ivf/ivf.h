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
#include "pca.h"

class IVF {
public:
    size_t N;
    size_t D;
    size_t C;
    size_t d; // the dimensionality of first a few dimensions

    float *L1_data;
    float *res_data;
    float *centroids;

    size_t *start;
    size_t *len;
    size_t *id;
    Index_PCA::PCA *PCA;

    IVF();

    IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive = 0);

    ~IVF();

    ResultHeap search(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    ResultHeap
    search_with_pca(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    ResultHeap search_with_learned_pca_lp(float *query, size_t k, size_t nprobe,
                                          float distK = std::numeric_limits<float>::max()) const;

    ResultHeap search_with_learned_pca_l2(float *query, size_t k, size_t nprobe,
                                          float distK = std::numeric_limits<float>::max()) const;

    std::vector<std::tuple<unsigned, float, float> > search_logger(float *query, size_t k, size_t nprobe) const;

    void compute_base_square(bool learned) const;

    void save(char *filename);

    void load(char *filename);
};

IVF::IVF() {
    N = D = C = d = 0;
    start = len = id = NULL;
    L1_data = res_data = centroids = NULL;
}

IVF::IVF(const Matrix<float> &X, const Matrix<float> &_centroids, int adaptive) {

    N = X.n;
    D = X.d;
    C = _centroids.n;

    assert(D > 32);
    start = new size_t[C];
    len = new size_t[C];
    id = new size_t[N];

    std::vector<size_t> *temp = new std::vector<size_t>[C];
    unsigned check_point = 0;
#pragma omp parallel for
    for (int i = 0; i < X.n; i++) {
        int belong = 0;
        float dist_min = X.dist(i, _centroids, 0);
        for (int j = 1; j < C; j++) {
            float dist = X.dist(i, _centroids, j);
            if (dist < dist_min) {
                dist_min = dist;
                belong = j;
            }
        }
#pragma omp critical
        {
            check_point++;
            if (check_point % 50000 == 0) {
                std::cerr << "Processing - " << check_point << " / " << X.n << std::endl;
            }
            temp[belong].push_back(i);
        }
    }
    std::cerr << "Cluster Generated!" << std::endl;

    size_t sum = 0;
    for (int i = 0; i < C; i++) {
        len[i] = temp[i].size();
        start[i] = sum;
        sum += len[i];
        for (int j = 0; j < len[i]; j++) {
            id[start[i] + j] = temp[i][j];
        }
    }

    if (adaptive == 1) d = 32;        // IVF++ - optimize cache (d = 32 by default)
    else if (adaptive == 0) d = D;   // IVF   - plain scan
    else d = 0;                     // IVF+  - plain ADSampling        

    L1_data = new float[N * d + 1];
    res_data = new float[N * (D - d) + 1];
    centroids = new float[C * D];
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int x = id[i];
        for (int j = 0; j < D; j++) {
            if (j < d) L1_data[i * d + j] = X.data[x * D + j];
            else res_data[i * (D - d) + j - d] = X.data[x * D + j];
        }
    }

    std::memcpy(centroids, _centroids.data, C * D * sizeof(float));
    delete[] temp;

}

IVF::~IVF() {
    if (id != NULL)delete[] id;
    if (len != NULL)delete[] len;
    if (start != NULL)delete[] start;
    if (L1_data != NULL)delete[] L1_data;
    if (res_data != NULL)delete[] res_data;
    if (centroids != NULL)delete[] centroids;
}

ResultHeap IVF::search(float *query, size_t k, size_t nprobe, float distK) const {
    // the default value of distK is +inf 
    Result *centroid_dist = new Result[C];

    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C; i++) {
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len[centroid_dist[i].second];
    if (d == D)adsampling::tot_dimension += 1ll * ncan * D;
    float *dist = new float[ncan];
    Result *candidates = new Result[ncan];
    int *obj = new int[ncan];

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D. 
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32). 
    int cur = 0;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = sqr_dist(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if (d > 0)dist[cur] = tmp_dist;
            else dist[cur] = 0;
            obj[cur] = can;
            cur++;
        }
    }
    ResultHeap KNNs;

    // d == D indicates FDScanning. 
    if (d == D) {
        for (int i = 0; i < ncan; i++) {
            candidates[i].first = dist[i];
            candidates[i].second = id[obj[i]];
        }
        std::partial_sort(candidates, candidates + k, candidates + ncan);

        for (int i = 0; i < k; i++) {
            KNNs.emplace(candidates[i].first, candidates[i].second);
        }
    }
    // d < D indicates ADSampling with and without cache-level optimization
    if (d < D) {
        auto cur_dist = dist;
        for (int i = 0; i < nprobe; i++) {
            int cluster_id = centroid_dist[i].second;
            for (int j = 0; j < len[cluster_id]; j++) {
                size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                float tmp_dist = adsampling::dist_comp(distK, res_data + can * (D - d), query + d, *cur_dist, d);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                if (tmp_dist > 0) {
                    KNNs.emplace(tmp_dist, id[can]);
                    if (KNNs.size() > k) KNNs.pop();
                }
                if (KNNs.size() == k && KNNs.top().first < distK) {
                    distK = KNNs.top().first;
                }
                cur_dist++;
            }
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}


ResultHeap IVF::search_with_pca(float *query, size_t k, size_t nprobe, float distK) const {
    // the default value of distK is +inf
    Result *centroid_dist = new Result[C];
    PCA->get_query_square(query);
    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C; i++) {
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len[centroid_dist[i].second];
    float *dist = new float[ncan];
    Result *candidates = new Result[ncan];
    int *obj = new int[ncan];

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32).
    int cur = 0;;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = PCA->get_pre_sum(can) - 2 * naive_lp_dist_calc(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if (d > 0)dist[cur] = tmp_dist;
            else dist[cur] = 0;
            obj[cur] = can;
            cur++;
        }
    }
    ResultHeap KNNs;

    // d < D indicates ADSampling with and without cache-level optimization
    if (d < D) {
        auto cur_dist = dist;
        for (int i = 0; i < nprobe; i++) {
            int cluster_id = centroid_dist[i].second;
            for (int j = 0; j < len[cluster_id]; j++) {
                size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                float tmp_dist = PCA->uniform_fast_inference(query + d, res_data + can * (D - d), distK, d,
                                                             *cur_dist);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                if (tmp_dist > 0) {
                    KNNs.emplace(tmp_dist, id[can]);
                    if (KNNs.size() > k) KNNs.pop();
                }
                if (KNNs.size() == k && KNNs.top().first < distK) {
                    distK = KNNs.top().first;
                }
                cur_dist++;
            }
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}


ResultHeap IVF::search_with_learned_pca_lp(float *query, size_t k, size_t nprobe, float distK) const {
    // the default value of distK is +inf and IVF++
    Result *centroid_dist = new Result[C];
    PCA->get_query_square(query);
    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C; i++) {
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len[centroid_dist[i].second];
    float *dist = new float[ncan];
    Result *candidates = new Result[ncan];
    int *obj = new int[ncan];

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32).
    int cur = 0;;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = PCA->get_pre_sum(can) - 2 * naive_lp_dist_calc(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            dist[cur] = tmp_dist;
            obj[cur] = can;
            cur++;
        }
    }
    ResultHeap KNNs;

    // d < D indicates ADSampling with and without cache-level optimization
    auto cur_dist = dist;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = PCA->learned_fast_inference_lp(query + d, res_data + can * (D - d), distK, 1,
                                                            *cur_dist);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if (tmp_dist > 0) {
                KNNs.emplace(tmp_dist, id[can]);
                if (KNNs.size() > k) KNNs.pop();
            }
            if (KNNs.size() == k && KNNs.top().first < distK) {
                distK = KNNs.top().first;
            }
            cur_dist++;
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}

ResultHeap IVF::search_with_learned_pca_l2(float *query, size_t k, size_t nprobe, float distK) const {
    // the default value of distK is +inf and IVF++
    Result *centroid_dist = new Result[C];
    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C; i++) {
#ifdef COUNT_DIST_TIME
        StopW stopw = StopW();
#endif
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
#ifdef COUNT_DIST_TIME
        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
        centroid_dist[i].second = i;
    }

    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len[centroid_dist[i].second];
    float *dist = new float[ncan];
    Result *candidates = new Result[ncan];
    int *obj = new int[ncan];

    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    // For IVF+ (i.e., apply ADSampling without optimizing data layout), it should be 0.
    // For IVF++ (i.e., apply ADSampling with optimizing data layout), it should be delta_d (i.e., 32).
    int cur = 0;;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = sqr_dist(query, L1_data + can * d, d);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            dist[cur] = tmp_dist;
            obj[cur] = can;
            cur++;
        }
    }
    ResultHeap KNNs;

    // d < D indicates ADSampling with and without cache-level optimization
    auto cur_dist = dist;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            float tmp_dist = PCA->learned_fast_inference_l2(query + d, res_data + can * (D - d), distK, 1,
                                                            *cur_dist);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            if (tmp_dist > 0) {
                KNNs.emplace(tmp_dist, id[can]);
                if (KNNs.size() > k) KNNs.pop();
            }
            if (KNNs.size() == k && KNNs.top().first < distK) {
                distK = KNNs.top().first;
            }
            cur_dist++;
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}


std::vector<std::tuple<unsigned, float, float> > IVF::search_logger(float *query, size_t k, size_t nprobe) const {
    // the default value of distK is +inf
    auto *centroid_dist = new Result[C];
    std::vector<std::tuple<unsigned, float, float> > search_logger;
    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C; i++) {
        centroid_dist[i].first = sqr_dist(query, centroids + i * D, D);
        centroid_dist[i].second = i;
    }
    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len[centroid_dist[i].second];
    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    std::priority_queue<float> res_queue;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len[cluster_id]; j++) {
            size_t can = start[cluster_id] + j;
            float tmp_dist;
            if (d == D)
                tmp_dist = sse_l2_dist_calc(query, L1_data + can * d, d);
            else
                tmp_dist = sse_l2_dist_calc(query, res_data + can * D, D);
            if (res_queue.size() < k) res_queue.push(tmp_dist);
            else if (tmp_dist < res_queue.top()) {
                res_queue.push(tmp_dist);
                if (res_queue.size() > k) res_queue.pop();
            }
            if (res_queue.size() == k)
                search_logger.emplace_back(can, tmp_dist, res_queue.top());
        }
    }
    delete[] centroid_dist;
    return search_logger;
}


void IVF::save(char *filename) {
    std::ofstream output(filename, std::ios::binary);

    output.write((char *) &N, sizeof(size_t));
    output.write((char *) &D, sizeof(size_t));
    output.write((char *) &C, sizeof(size_t));
    output.write((char *) &d, sizeof(size_t));

    if (d > 0)output.write((char *) L1_data, N * d * sizeof(float));
    if (d < D)output.write((char *) res_data, N * (D - d) * sizeof(float));
    output.write((char *) centroids, C * D * sizeof(float));

    output.write((char *) start, C * sizeof(size_t));
    output.write((char *) len, C * sizeof(size_t));
    output.write((char *) id, N * sizeof(size_t));

    output.close();
}

void IVF::load(char *filename) {
    std::ifstream input(filename, std::ios::binary);
    cerr << filename << endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.read((char *) &N, sizeof(size_t));
    input.read((char *) &D, sizeof(size_t));
    input.read((char *) &C, sizeof(size_t));
    input.read((char *) &d, sizeof(size_t));
    cerr << N << " " << D << " " << C << " " << d << endl;

    L1_data = new float[N * d + 10];
    res_data = new float[N * (D - d) + 10];
    centroids = new float[C * D];

    start = new size_t[C];
    len = new size_t[C];
    id = new size_t[N];

    if (d > 0)input.read((char *) L1_data, N * d * sizeof(float));
    if (d < D)input.read((char *) res_data, N * (D - d) * sizeof(float));
    input.read((char *) centroids, C * D * sizeof(float));

    input.read((char *) start, C * sizeof(size_t));
    input.read((char *) len, C * sizeof(size_t));
    input.read((char *) id, N * sizeof(size_t));

    input.close();
}


void IVF::compute_base_square(bool learned) const {
    PCA->base_square = new float[N];
    float *extra = PCA->extra_mean;
    float *tmp_L1 = L1_data, *tmp_res = res_data;
    for (int i = 0; i < N; i++) {
        float square = 0.0;
        for (int j = 0; j < d; j++) {
            if (!learned) tmp_L1[j] -= extra[j];
            square += tmp_L1[j] * tmp_L1[j];
        }
        for (int j = 0; j < D - d; j++) {
            if (!learned) tmp_res[j] -= extra[j + d];
            square += tmp_res[j] * tmp_res[j];
        }
        PCA->base_square[i] = square;
        tmp_L1 += d;
        tmp_res += (D - d);
    }
    if (!learned) {
        float *tmp_centroid = centroids;
        for (int i = 0; i < C; i++) {
            for (int j = 0; j < D; j++) {
                tmp_centroid[j] -= extra[j];
            }
            tmp_centroid += D;
        }
    }
}