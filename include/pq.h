//
// Created by mingyu on 23-7-20.
//
#include "utils.h"
#include "adsampling.h"

#ifndef LEARN_TO_PRUNE_PQ_H
#define LEARN_TO_PRUNE_PQ_H
#define AVX_SZ 8
#define SSE_SZ 4

namespace Index_PQ {
    class Quantizer {
    public:
        Quantizer(unsigned num, unsigned dimension) {
            nd_ = num;
            dimension_ = dimension;
        }
        /*
         * PQ data
         */
        // format i-th sub-vector j-th sub-vector-cluster_id k-th cluster-data
        typedef std::vector<std::vector<std::vector<float> > > CodeBook;
        CodeBook pq_book;
        float *dist_mp;
        unsigned sub_dim, sub_vector, sub_cluster_count;
        unsigned dimension_, nd_;
        uint8_t *pq_mp;
        float *node_cluster_dist_;
        Eigen::MatrixXf A_;

        std::vector<float> W_, B_, b_;
        double eps = 1e-5, recall = 0.995;
        unsigned count_base, model_count;
        bool verbose = true;

        void load_linear_model(const char *filename) {
            if (!isFileExists_ifstream(filename)) return;
            std::ifstream fin(filename);
            unsigned num;
            fin >> num;
            model_count = num;
            W_.resize(2);
            B_.resize(1);
            fin >> W_[0] >> W_[1] >> B_[0];
            fin.close();
        }

        __attribute__((always_inline))
        inline bool linear_classifier_default_pq(float app_dist, float cluster_dist, float thresh_dist) {
            return app_dist * W_[0] + cluster_dist * W_[1] + B_[0] > thresh_dist;
        }

        void load_product_codebook(const char *filename) {
            std::ifstream in(filename, std::ios::binary);
            in.read((char *) &sub_vector, sizeof(unsigned));
            in.read((char *) &sub_cluster_count, sizeof(unsigned));
            in.read((char *) &sub_dim, sizeof(unsigned));
            std::cerr << "sub vec:: " << sub_vector << " sub cluster:: " << sub_cluster_count << " sub dim:: "
                      << sub_dim << std::endl;
            pq_book.resize(sub_vector);
            dist_mp = new float[sub_vector * sub_cluster_count];
            for (int i = 0; i < sub_vector; i++) {
                pq_book[i].resize(sub_cluster_count);
                for (int j = 0; j < sub_cluster_count; j++) {
                    pq_book[i][j].resize(sub_dim);
                    in.read((char *) pq_book[i][j].data(), sizeof(float) * sub_dim);
                }
            }
        }

        void calc_dist_map(float *query) {
            for (unsigned i = 0; i < sub_vector; i++) {
                for (unsigned j = 0; j < sub_cluster_count; j++) {
                    dist_mp[i * sub_cluster_count + j] = naive_l2_dist_calc(query + i * sub_dim, &pq_book[i][j][0],
                                                                            sub_dim);
                }
            }
        }

        float naive_product_map_dist(unsigned id) const {
            float res = 0;
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += sub_vector;
#endif
            for (int i = 0; i < sub_vector; i++) {
                res += dist_mp[i * sub_cluster_count + pq_mp[id * sub_vector + i]];
            }
            return res;
        }

        __attribute__((always_inline))
        inline void sse4_product_map_dist(const uint8_t *const pqcode0,
                                          const uint8_t *const &pqcode1,
                                          const uint8_t *const &pqcode2,
                                          const uint8_t *const &pqcode3,
                                          float *&dists,
                                          __m128 &candidates
        ) const {
            candidates = _mm_set_ps(
                    dists[pqcode3[0]],
                    dists[pqcode2[0]],
                    dists[pqcode1[0]],
                    dists[pqcode0[0]]
            );
            // Such perf critical loop. Pls unroll
            for (unsigned j = 1; j < sub_vector; ++j) {
                const float *const cdist = dists + j * sub_cluster_count;
                __m128 partial = _mm_set_ps(
                        cdist[pqcode3[j]],
                        cdist[pqcode2[j]],
                        cdist[pqcode1[j]],
                        cdist[pqcode0[j]]
                );
                candidates = _mm_add_ps(candidates, partial);
            }
        }

        /** Base functions for avx **/
        __attribute__((always_inline))
        inline void axv8_product_map_dist(const uint8_t *const pqcode0,
                                          const uint8_t *const &pqcode1,
                                          const uint8_t *const &pqcode2,
                                          const uint8_t *const &pqcode3,
                                          const uint8_t *const &pqcode4,
                                          const uint8_t *const &pqcode5,
                                          const uint8_t *const &pqcode6,
                                          const uint8_t *const &pqcode7,
                                          float *&dists,
                                          __m256 &candidates
        ) const {
            candidates = _mm256_set_ps(
                    dists[pqcode7[0]],
                    dists[pqcode6[0]],
                    dists[pqcode5[0]],
                    dists[pqcode4[0]],
                    dists[pqcode3[0]],
                    dists[pqcode2[0]],
                    dists[pqcode1[0]],
                    dists[pqcode0[0]]
            );
            // Such perf critical loop. Pls unroll
            for (unsigned j = 1; j < sub_vector; ++j) {
                const float *const cdist = dists + j * sub_cluster_count;
                __m256 partial = _mm256_set_ps(
                        cdist[pqcode7[j]],
                        cdist[pqcode6[j]],
                        cdist[pqcode5[j]],
                        cdist[pqcode4[j]],
                        cdist[pqcode3[j]],
                        cdist[pqcode2[j]],
                        cdist[pqcode1[j]],
                        cdist[pqcode0[j]]
                );
                candidates = _mm256_add_ps(candidates, partial);
            }
        }

        float *sse4_dist_sacn(const unsigned *id, unsigned num) {
            auto res = new float[num];
            float arr[4];
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += sub_vector * num;
#endif
            for (int i = 0; i < num; i += 4) {
                __m128 candidate_dist;
                const uint8_t *const pqcode0 = pq_mp + id[i] * sub_vector;
                const uint8_t *const pqcode1 = pq_mp + id[i + 1] * sub_vector;
                const uint8_t *const pqcode2 = pq_mp + id[i + 2] * sub_vector;
                const uint8_t *const pqcode3 = pq_mp + id[i + 3] * sub_vector;
                sse4_product_map_dist(pqcode0, pqcode1, pqcode2, pqcode3, dist_mp, candidate_dist);
                _mm_store_ps(arr, candidate_dist);
                memcpy(res + i, arr, 16);
            }
            return res;
        }


        float *avx8_dist_sacn(const unsigned *id, unsigned num) {
            auto res = new float[num];
            float arr[8];
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += sub_vector * num;
#endif
            for (int i = 0; i < num; i += 8) {
                __m256 candidate_dist;
                const uint8_t *const pqcode0 = pq_mp + id[i] * sub_vector;
                const uint8_t *const pqcode1 = pq_mp + id[i + 1] * sub_vector;
                const uint8_t *const pqcode2 = pq_mp + id[i + 2] * sub_vector;
                const uint8_t *const pqcode3 = pq_mp + id[i + 3] * sub_vector;
                const uint8_t *const pqcode4 = pq_mp + id[i + 4] * sub_vector;
                const uint8_t *const pqcode5 = pq_mp + id[i + 5] * sub_vector;
                const uint8_t *const pqcode6 = pq_mp + id[i + 6] * sub_vector;
                const uint8_t *const pqcode7 = pq_mp + id[i + 7] * sub_vector;
                axv8_product_map_dist(pqcode0, pqcode1, pqcode2, pqcode3, pqcode4, pqcode5, pqcode6, pqcode7, dist_mp,
                                      candidate_dist);
                _mm256_store_ps(arr, candidate_dist);
                memcpy(res + i, arr, 32);
            }
            return res;
        }


        float naive_product_dist(unsigned id, const float *query) const {
            float res = 0;
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += sub_vector;
#endif
            for (int i = 0; i < sub_vector; i++) {
                res += naive_l2_dist_calc(query + i * sub_dim, &pq_book[i][pq_mp[id * sub_vector + i]][0], sub_dim);
            }
            return res;
        }


        void load_project_matrix(const char *filename) {
            float *raw_data;
            unsigned origin_dim, project_dim;
            load_float_data(filename, raw_data, origin_dim, project_dim);
            A_ = Eigen::MatrixXf(origin_dim, project_dim);
            for (int i = 0; i < origin_dim; i++) {
                for (int j = 0; j < project_dim; j++) {
                    A_(i, j) = raw_data[i * project_dim + j]; // load the matrix
                }
            }
        }

        void project_vector(float *raw_data, unsigned num) const {
            Eigen::MatrixXf Q(num, dimension_);
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dimension_; j++) {
                    Q(i, j) = raw_data[i * dimension_ + j];
                }
            }
            Q = Q * A_;
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dimension_; j++) {
                    raw_data[i * dimension_ + j] = Q(i, j);
                }
            }
        }

        void binary_search_multi_linear(unsigned num, const float *app_dist, const float *acc_dist,
                                        const float *cluster_dist, const float *thresh) {
            double l = 0.0, r = 0.0, res;
            for (int i = 0; i < num; i++) {
                if (thresh[i] > r) r = thresh[i];
            }
            l = -r;
            std::cerr << l << " " << r <<" "<<W_[0]<<" "<<W_[1]<<endl;
            while (r - l > eps) {
                double mid = (l + r) / 2.0;
                unsigned bad_count = 0;
#pragma omp parallel for reduction(+:bad_count)
                for (int i = 0; i < num; i++) {
                    if (app_dist[i] * W_[0] + cluster_dist[i] * W_[1] + mid > thresh[i] && (double) acc_dist[i] < (double) thresh[i] + 1e-6) {
                        bad_count++;
                    }
                }
                double test_recall = (double) ((double)count_base - (double) bad_count) / (double) count_base;
                if (test_recall < recall) {
                    r = mid - eps;
                } else {
                    std::cerr << mid << " <-gap-> " << r << " recall::" << test_recall<<" bad-> "<<bad_count<< endl;
                    res = mid;
                    l = mid + eps;
                }
            }
            B_[0] = (float) res;
        }


    };
}


#endif //LEARN_TO_PRUNE_PQ_H