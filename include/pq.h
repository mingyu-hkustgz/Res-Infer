//
// Created by mingyu on 23-7-20.
//
#include "utils.h"
#ifndef LEARN_TO_PRUNE_PQ_H
#define LEARN_TO_PRUNE_PQ_H
#define AVX_SZ 8
#define SSE_SZ 4

namespace Index_PQ {
    class Quantizer {
    public:
        Quantizer(unsigned num, unsigned dimension, float *data) {
            data_ = data;
            nd_ = num;
            dimension_ = dimension;
        }
        /*
         * PQ data
         */
        // format i-th sub-vector j-th sub-vector-cluster_id k-th cluster-data
        typedef std::vector<std::vector<std::vector<float> > > CodeBook;
        CodeBook pq_book;
        float* dist_mp;
        unsigned sub_dim, sub_vector, sub_cluster_count;
        unsigned dimension_, nd_;
        uint8_t *pq_mp;
        float *node_cluster_dist_;
        float *data_;
        Eigen::MatrixXf A_;


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

        void encoder_origin_data() {
            pq_mp = new unsigned char[nd_ * sub_vector];
            node_cluster_dist_ = new float[nd_];
            double ave_dist = 0.0;
#pragma omp parallel for
            for (int i = 0; i < nd_; i++) {
                float dist_to_centroid = 0.0;
                for (int j = 0; j < sub_vector; j++) {
                    uint8_t belong = 0;
                    float dist = naive_l2_dist_calc(data_ + i * dimension_, pq_book[j][0].data(), sub_dim);
                    for (int k = 1; k < sub_cluster_count; k++) {
                        float new_dist = naive_l2_dist_calc(data_ + i * dimension_ + j * sub_dim, pq_book[j][k].data(),
                                                            sub_dim);
                        if (new_dist < dist) {
                            belong = k;
                            dist = new_dist;
                        }
                    }
                    dist_to_centroid += dist;
                    pq_mp[i * sub_vector + j] = belong;
                }
                node_cluster_dist_[i] = dist_to_centroid;
#pragma omp critical
                ave_dist += dist_to_centroid;
            }
            std::cerr << "Encoder ave dist:: " << ave_dist / nd_ << std::endl;
        }

        void calc_dist_map(const float *query) {
            for (unsigned i = 0; i < sub_vector; i++) {
                for (unsigned j = 0; j < sub_cluster_count; j++) {
                    dist_mp[i * sub_cluster_count + j] = naive_l2_dist_calc(query + i * sub_dim, &pq_book[i][j][0],
                                                                            sub_dim);
                }
            }
        }

        float naive_product_map_dist(unsigned id) const {
            float res = 0;
            for (int i = 0; i < sub_vector; i++) {
                res += dist_mp[i * sub_cluster_count +  pq_mp[id * sub_vector + i]];
            }
            return res;
        }

        __attribute__((always_inline))
        inline void sse4_product_map_dist(const uint8_t* const pqcode0,
                                          const uint8_t* const& pqcode1,
                                          const uint8_t* const& pqcode2,
                                          const uint8_t* const& pqcode3,
                                          float*& dists,
                                          __m128& candidates
        ) const {
            candidates = _mm_set_ps(
                    dists[pqcode3[0]],
                    dists[pqcode2[0]],
                    dists[pqcode1[0]],
                    dists[pqcode0[0]]
            );
            // Such perf critical loop. Pls unroll
            for (unsigned j = 1; j < sub_vector; ++j) {
                const float* const cdist = dists + j * sub_cluster_count;
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
        inline void axv8_product_map_dist(const uint8_t* const pqcode0,
                                               const uint8_t* const& pqcode1,
                                               const uint8_t* const& pqcode2,
                                               const uint8_t* const& pqcode3,
                                               const uint8_t* const& pqcode4,
                                               const uint8_t* const& pqcode5,
                                               const uint8_t* const& pqcode6,
                                               const uint8_t* const& pqcode7,
                                               float*& dists,
                                               __m256& candidates
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
                const float* const cdist = dists + j * sub_cluster_count;
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



        float naive_product_dist(unsigned id, const float *query) const {
            float res = 0;
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
    };
}


#endif //LEARN_TO_PRUNE_PQ_H
