//
// Created by BLD on 23-7-28.
//
#include "utils.h"
#include "adsampling.h"

#ifndef LEARN_TO_PRUNE_LINEAR_H
#define LEARN_TO_PRUNE_LINEAR_H
namespace Linear {
    struct Linear {
        Linear(unsigned dimension) {
            dim = dimension;
        }

        __attribute__((always_inline))
        inline bool linear_classifier_default_pq(float app_dist, float cluster_dist, float thresh_dist) {
            if (!model_count) return app_dist > thresh_dist;
            else return app_dist * W_[0] + cluster_dist * W_[1] + B_[0] > thresh_dist;
        }

        __attribute__((always_inline))
        inline bool linear_classifier_ratio_only(float app_dist, float thresh_dist) {
            return app_dist * W_[0] > thresh_dist;
        }

        __attribute__((always_inline))
        inline bool linear_classifier(float app_dist, float thresh_dist) {
            return app_dist * W_[0] + B_[0] > thresh_dist;
        }

        __attribute__((always_inline))
        inline bool target_linear_classifier_(float app_dist, float thresh_dist, unsigned num) {
            return app_dist * W_[num] + B_[num] > thresh_dist;
        }

        __attribute__((always_inline))
        inline float
        multi_linear_classifier_(float *q, float *p, float thresh_dist, unsigned tag_model = 0, float res = 0) {
            unsigned cur = origin_dim;
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += origin_dim;
#endif
            if (tag_model == 1) {
                if (target_linear_classifier_(res, thresh_dist, 0)) return -res * W_[0] - b_[0];
                cur += origin_dim;
            }
            for (; cur <= fix_dim; cur += origin_dim) {
                res += sqr_dist(p, q, origin_dim);
                p += origin_dim;
                q += origin_dim;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += origin_dim;
#endif
                if (target_linear_classifier_(res, thresh_dist, tag_model)) return -res * W_[tag_model] - b_[tag_model];
                tag_model++;
            }
            if (res_dim) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += res_dim;
#endif
                res += sqr_dist(q, p, res_dim);
                if (res > thresh_dist) return -res;
            }
            return res;
        }

        std::vector<float> W_, B_, b_;

        void load_linear_model(const char *filename) {
            if (!isFileExists_ifstream(filename)) return;
            std::ifstream fin(filename);
            unsigned num;
            fin >> num;
            if (num != 1) {
                W_.resize(num);
                B_.resize(num);
                b_.resize(num);
                model_count = num;
                for (int i = 0; i < num; i++) {
                    fin >> W_[i] >> B_[i] >> b_[i];
                }
            } else {
                model_count = num;
                W_.resize(2);
                B_.resize(1);
                fin >> W_[0] >> W_[1] >> B_[0];
            }
            res_dim = dim % origin_dim;
            fix_dim = dim - res_dim;
            std::cerr << fix_dim << " " << res_dim << " " << num << std::endl;
            fin.close();
        }

        void
        binary_search_single_linear(unsigned num, const float *app_dist, const float *acc_dist, const float *thresh,
                                    unsigned id) {
            double l = 0.0, r = 0.0, res;
            for (int i = 0; i < num; i++) {
                if (thresh[i] * W_[id] > r) r = thresh[i] * W_[id] * 1.01;
            }
            l = -r;
            std::cerr << l << " " << r << endl;
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
                    std::cerr << mid << " <-gap-> " << r << " recall::" << test_recall << " bad-> " << bad_count
                              << endl;
                    res = mid;
                    l = mid + eps;
                }
            }
            B_[id] = (float) res;
        }

        void binary_search_multi_linear(unsigned num, const float *app_dist, const float *acc_dist,
                                        const float *cluster_dist, const float *thresh) {
            double l = 0.0, r = 0.0, res;
            for (int i = 0; i < num; i++) {
                if (thresh[i] > r) r = thresh[i];
            }
            l = -r;
            std::cerr << l << " " << r << " " << W_[0] << " " << W_[1] << endl;
            while (r - l > eps) {
                double mid = (l + r) / 2.0;
                unsigned bad_count = 0;
#pragma omp parallel for reduction(+:bad_count)
                for (int i = 0; i < num; i++) {
                    if ((double)app_dist[i] * W_[0] + cluster_dist[i] * W_[1] + mid > (double)thresh[i] &&
                        (double) acc_dist[i] < (double) thresh[i] + 1e-6) {
                        bad_count++;
                    }
                }
                double test_recall = (double) ((double) count_base - (double) bad_count) / (double) count_base;
                if (test_recall < recall) {
                    r = mid - eps;
                } else {
                    std::cerr << mid << " <-gap-> " << r << " recall::" << test_recall << " bad-> " << bad_count
                              << endl;
                    res = mid;
                    l = mid + eps;
                }
            }
            B_[0] = (float) res;
        }

        double eps = 0.0001;
        double recall = 0;
        unsigned origin_dim = 32;
        unsigned count_base;
        unsigned model_count = 0, dim, res_dim, fix_dim;
    };

}

#endif //LEARN_TO_PRUNE_LINEAR_H
