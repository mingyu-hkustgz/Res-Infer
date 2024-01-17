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


        void load_linear_model(char *filename) {
            if (!isFileExists_ifstream(filename)) return;
            std::ifstream fin(filename);
            unsigned num;
            fin >> num;
            W_.resize(num);
            B_.resize(num);
            b_.resize(num);
            model_count = num;
            for (int i = 0; i < num; i++) {
                fin >> W_[i] >> B_[i] >> b_[i];
            }
            learn_res_dim = dimension_ % base_dim;
            fix_dim = dimension_ - learn_res_dim;
            std::cerr << " fix dim:: " << fix_dim << " res dim:: " << learn_res_dim << " linear count:: " << num
                      << std::endl;
            fin.close();
        }


        /*
         * note that we do not add feature normalization
         */
        void project_vector(float *raw_data, unsigned num, bool learned = false) const {
            Eigen::MatrixXf Q(num, dimension_);
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dimension_; j++) {
                    Q(i, j) = raw_data[i * dimension_ + j] - mean_[j];
                }
            }
            Q = Q * X_;
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dimension_; j++) {
                    if (!learned)
                        raw_data[i * dimension_ + j] = Q(i, j) - extra_mean[j];
                    else
                        raw_data[i * dimension_ + j] = Q(i, j);
                }
            }
        }

        void load_project_matrix(const char *filename) {
            float *raw_data;
            unsigned hybrid, project_dim;
            load_float_data(filename, raw_data, hybrid, project_dim);
            unsigned origin_dim = hybrid - 3;
            std::cerr << "origin dim:: " << origin_dim << std::endl;
            dimension_ = origin_dim;
            mean_ = new float[origin_dim];
            extra_mean = new float[origin_dim];
            var = new float[origin_dim];
            pre_query = new float[origin_dim + 1];
            for (int i = 0; i < origin_dim; i++) mean_[i] = raw_data[i];
            raw_data = raw_data + origin_dim;
            for (int i = 0; i < origin_dim; i++) extra_mean[i] = raw_data[i];
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
                fix_dim = dimension_ - uni_res_dim;
            }
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
        inline bool uniform_inference(float &res, float &threshold, int &tag_dim) const {
            if (res - pre_query[tag_dim] > threshold) return true;
            else return false;
        }

        __attribute__((always_inline))
        inline bool learned_inference(float &res, float &threshold, int tag_model) const {
            return res * W_[tag_model] + B_[tag_model] > threshold;
        }


        __attribute__((always_inline))
        inline float get_pre_sum(unsigned id) const {
            float res = base_square[id] + query_square + 1e-5;
            return res;
        }

        __attribute__((always_inline))
        inline float uniform_fast_inference(float *q, float *p, float threshold, int tag_dim = 0, float res = 0) const {
            int cur = tag_dim;
            if (cur != 0) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
                if (uniform_inference(res, threshold, cur)) return -res;
            }
            while (cur < fix_dim) {
                res -= 2 * naive_lp_dist_calc(p, q, base_dim);
                p += base_dim;
                q += base_dim;
                cur += base_dim;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
                if (uniform_inference(res, threshold, cur)) return -res;
            }
            if (uni_res_dim) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += uni_res_dim;
#endif
                res -= 2 * naive_lp_dist_calc(q, p, uni_res_dim);
            }
            if (res > threshold) return -res;
            else return res;
        }


        __attribute__((always_inline))
        inline float
        learned_fast_inference_l2(float *q, float *p, float thresh_dist, int tag_model = 0, float res = 0) {
            unsigned cur = base_dim;
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += base_dim;
#endif
            if (tag_model == 1) {
                if (learned_inference(res, thresh_dist, 0)) return -res * W_[0] - b_[0];
                cur += base_dim;
            }
            for (; cur <= fix_dim; cur += base_dim) {
                res += sqr_dist(p, q, base_dim);
                p += base_dim;
                q += base_dim;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
                if (learned_inference(res, thresh_dist, tag_model)) return -res * W_[tag_model] - b_[tag_model];
                tag_model++;
            }
            if (learn_res_dim) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += learn_res_dim;
#endif
                res += sqr_dist(q, p, learn_res_dim);
                if (res > thresh_dist) return -res;
            }
            return res;
        }

        __attribute__((always_inline))
        inline float
        learned_fast_inference_lp(float *q, float *p, float thresh_dist, int tag_model = 0, float res = 0) {
            unsigned cur = base_dim;
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += base_dim;
#endif
            if (tag_model == 1) {
                if (learned_inference(res, thresh_dist, 0)) return -res * W_[0] - b_[0];
                cur += base_dim;
            }
            for (; cur <= fix_dim; cur += base_dim) {
                res -= 2 * naive_lp_dist_calc(p, q, base_dim);
                p += base_dim;
                q += base_dim;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += base_dim;
#endif
                if (learned_inference(res, thresh_dist, tag_model)) return -res * W_[tag_model] - b_[tag_model];
                tag_model++;
            }
            if (learn_res_dim) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += learn_res_dim;
#endif
                res -= 2 * naive_lp_dist_calc(p, q, learn_res_dim);
                if (res > thresh_dist) return -res;
            }
            return res;
        }


        void
        binary_search_single_linear(unsigned num, const float *app_dist, const float *acc_dist, const float *thresh,
                                    unsigned id) {
            double l = 0.0, r = 0.0, res;
            for (int i = 0; i < num; i++) {
                if (thresh[i] * W_[id] > r) r = thresh[i] * W_[id] * 1.01;
            }
            l = -r;
            if (verbose)
                std::cerr << l << " <-left right-> " << r << endl;
            while (r - l > eps) {
                double mid = (l + r) / 2.0;
                unsigned bad_count = 0;
#pragma omp parallel for reduction(+:bad_count)
                for (int i = 0; i < num; i++) {
                    if (app_dist[i] * W_[id] + mid > thresh[i] && (double) acc_dist[i] < (double) thresh[i] + 1e-6) {
                        bad_count++;
                    }
                }
                bad_count = std::min(bad_count, count_base);
                double test_recall = (double) ((double) count_base - (double) bad_count) / (double) count_base;
                if (test_recall < recall) {
                    r = mid - eps;
                } else {
                    if (verbose)
                        std::cerr << mid << " <-gap-> " << r << " recall::" << test_recall << " bad-> " << bad_count
                                  << endl;
                    res = mid;
                    l = mid + eps;
                }
            }
            B_[id] = (float) res;
        }


        Eigen::MatrixXf X_;
        Eigen::MatrixXd vec, val;
        int base_dim = 32, dimension_, uni_res_dim, learn_res_dim, fix_dim;
        unsigned nd_;
        float *mean_ = nullptr, *extra_mean = nullptr, *var = nullptr, *base_square = nullptr, *pre_query = nullptr;
        float sigma_count = 3.0, query_square = 0;

        std::vector<float> W_, B_, b_;
        double eps = 1e-5, recall = 0.995;
        unsigned count_base, model_count;
        bool verbose = true;
    };

}


#endif //LEARN_TO_PRUNE_PCA_H
