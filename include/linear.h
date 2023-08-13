//
// Created by BLD on 23-7-28.
//
#include "utils.h"

#ifndef LEARN_TO_PRUNE_LINEAR_H
#define LEARN_TO_PRUNE_LINEAR_H
namespace Linear {
    struct Linear {
        Linear(unsigned dimension) {
            dim = dimension;
        }

        __attribute__((always_inline))
        inline bool linear_classifier_default_pq(float app_dist, float thresh_dist) {
            return app_dist * W_[0] + B_[0] > thresh_dist;
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
        inline float multi_linear_classifier_(float *q, float *p, float thresh_dist, unsigned tag_model = 0, float res = 0) {
            unsigned cur = origin_dim;
            if(tag_model==1){
                if (target_linear_classifier_(res, thresh_dist, 0)) return -res * W_[0] + B_[0];
                cur += origin_dim;
            }
            for (; cur <= dim; cur += origin_dim) {
                res += naive_l2_dist_calc(q, p, origin_dim);
                p += origin_dim;
                q += origin_dim;
                if (target_linear_classifier_(res, thresh_dist, tag_model)) return -res * W_[tag_model] + B_[tag_model];
                tag_model++;
            }
            return res;
        }

        std::vector<float> W_, B_;

        void load_linear_model(const char *filename) {
            std::ifstream fin(filename);
            unsigned num;
            fin >> num;
            W_.resize(num);
            B_.resize(num);
            model_count = num;
            for (int i = 0; i < num; i++) {
                fin >> W_[i] >> B_[i];
            }
            origin_dim = dim / model_count;
        }

        void
        binary_search_single_linear(unsigned num, const float *app_dist, const float *acc_dist, const float *thresh,
                                    unsigned id) {
            double l = 0.0, r = 0.0, res;
            for (int i = 0; i < num; i++) {
                if (thresh[i] > r) r = thresh[i];
            }
            l = -r;
            std::cerr << l << " " << r << endl;
            while (r - l > eps) {
                double mid = (l + r) / 2.0;
                unsigned bad_count = 0;
#pragma omp parallel for reduction(+:bad_count)
                for (int i = 0; i < num; i++) {
                    if (app_dist[i] * W_[id] + mid > thresh[i] && acc_dist[i] < thresh[i]) {
                        bad_count++;
                    }
                }
                double test_recall = (double) (num - bad_count) / (double) num;
                if (test_recall < recall) {
                    r = mid - eps;
                    res = mid;
                    std::cerr << l << " " << mid << " " << test_recall << endl;
                } else {
                    std::cerr << mid << " " << r << " " << test_recall << endl;
                    l = mid + eps;
                }
            }
            B_[id] = (float) res;
        }

        double eps = 0.0001;
        double recall = 0;
        unsigned model_count, origin_dim, dim;
    };

}

#endif //LEARN_TO_PRUNE_LINEAR_H
