//
// Created by BLD on 23-7-22.
//
#include "utils.h"
#include "adsampling.h"

#ifndef LEARN_TO_PRUNE_PCA_H
#define LEARN_TO_PRUNE_PCA_H
namespace Index_PCA {
    class PCA {
    public:
        PCA(unsigned num, unsigned dim) {
            nd_ = num;
            dimension_ = (int) dim;// the project dim
        }

        /*
         * note that we do not add feature normalization
         */
        void project_vector(float *raw_data, unsigned num) const {
            Eigen::MatrixXf Q(num, dimension_);
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dimension_; j++) {
                    Q(i, j) = raw_data[i * dimension_ + j] - mean_[j];
                }
            }
            Q = Q * X_;
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dimension_; j++) {
                    raw_data[i * dimension_ + j] = Q(i, j);
                }
            }
        }

        void load_project_matrix(const char *filename) {
            float *raw_data;
            unsigned hybrid, project_dim;
            load_float_data(filename, raw_data, hybrid, project_dim);
            unsigned origin_dim = hybrid - 2;
            std::cerr << "origin dim:: " << origin_dim << std::endl;
            dimension_ = origin_dim;
            mean_ = new float[origin_dim];
            var = new float[origin_dim];
            pre_query = new float[origin_dim + 1];
            for (int i = 0; i < origin_dim; i++) mean_[i] = raw_data[i];
            raw_data = raw_data + origin_dim;
            for (int i = 0; i < origin_dim; i++) var[i] = raw_data[i];
            float *matrix = raw_data + origin_dim;
            X_ = Eigen::MatrixXf(origin_dim, project_dim);
            for (int i = 0; i < origin_dim; i++) {
                for (int j = 0; j < project_dim; j++) {
                    X_(i, j) = matrix[i * project_dim + j]; // load the matrix form sklearn.decomposition
                }
            }
            if (base_dim) {
                uni_res_dim = dimension_ % base_dim;
                int cur_dim = base_dim;
                while (cur_dim <= dimension_) cur_dim = cur_dim + cur_dim;
                if (cur_dim > dimension_) cur_dim >>= 1;
                log_res_dim = dimension_ - cur_dim;
            }
        }

        void load_base_square(const char *filename) {
            unsigned points_num, dim;
            load_float_data(filename, base_square, points_num, dim);
            std::cerr << "square num:: " << dim << endl;
        }

        void get_query_square(const float *q) {
            query_square = 0;
            for (int i = 0; i < dimension_; i++) {
                query_square += q[i] * q[i];
                pre_query[i] = q[i] * q[i] * var[i];
            }
            pre_query[dimension_] = 0;
            for (int i = (int) dimension_ - 1; i >= 0; i--) {
                pre_query[i] += pre_query[i + 1];
            }
            for (int i = 0; i < dimension_; i++) {
                pre_query[i] = sqrt(pre_query[i]);
                pre_query[i] *= sigma_count * 2;
            }
        }

        __attribute__((always_inline))
        inline bool target_linear_classifier_(float res, float threshold, int tag_dim) const {
            if (res - pre_query[tag_dim] > threshold) return true;
            else return false;
        }

        __attribute__((always_inline))
        inline float get_pre_sum(unsigned id) const {
            float res = base_square[id] + query_square;
            return res;
        }

        __attribute__((always_inline))
        inline float uniform_fast_inference(float *q, float *p, float threshold, int tag_dim = 0, float res = 0) const {
            int cur = tag_dim;
            if (cur != 0) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
                if (target_linear_classifier_(res, threshold, cur)) return -res;
            }
            while (cur < dimension_) {
                res -= 2 * naive_lp_dist_calc(p, q, base_dim);
                p += base_dim;
                q += base_dim;
                cur += base_dim;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
                if (target_linear_classifier_(res, threshold, cur)) return -res;
            }
            if (uni_res_dim) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += uni_res_dim;
#endif
                res -= 2 * naive_lp_dist_calc(q, p, uni_res_dim);
                if (res > threshold) return -res;
            }
            return res;
        }

        __attribute__((always_inline))
        inline float log_fast_inference(float *q, float *p, float threshold, int tag_dim = 0, float res = 0) const {
            int cur = tag_dim;
            if (cur != 0) {
                if (target_linear_classifier_(res, threshold, cur)) return -res;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
            } else {
                cur = base_dim;
                res -= 2 * naive_lp_dist_calc(p, q, cur);
                p += cur;
                q += cur;
                if (target_linear_classifier_(res, threshold, cur)) return -res;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
            }
            while (cur << 1 <= dimension_) {
                res -= 2 * naive_lp_dist_calc(p, q, cur);
                p += cur;
                q += cur;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += cur;
#endif
                cur <<= 1;
                if (target_linear_classifier_(res, threshold, cur)) return -res;
            }
            if (log_res_dim) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += log_res_dim;
#endif
                res -= 2 * naive_lp_dist_calc(q, p, log_res_dim);
                if (res > threshold) return -res;
            }
            return res;
        }


        Eigen::MatrixXf X_;
        Eigen::MatrixXd vec, val;
        int base_dim = 32, dimension_, uni_res_dim, log_res_dim;
        unsigned nd_;
        float *mean_, *var, *base_square, *pre_query;
        float sigma_count = 3.0, query_square = 0;
    };

}


#endif //LEARN_TO_PRUNE_PCA_H
