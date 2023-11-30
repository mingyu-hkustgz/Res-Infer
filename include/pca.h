//
// Created by BLD on 23-7-22.
//
#include "utils.h"

#ifndef LEARN_TO_PRUNE_PCA_H
#define LEARN_TO_PRUNE_PCA_H
namespace Index_PCA{
    class PCA{
        public:
            PCA(unsigned num, unsigned dim){
                nd_ = num;
                dimension_ = dim;// the project dim
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
            unsigned origin_dim = hybrid-1;
            mean_ = new float[origin_dim];
            for(int i=0;i<origin_dim;i++) mean_[i] = raw_data[i];
            float *matrix = raw_data + origin_dim;
            X_ = Eigen::MatrixXf(origin_dim, project_dim);
            for (int i = 0; i < origin_dim; i++) {
                for (int j = 0; j < project_dim; j++) {
                    X_(i, j) = matrix[i * project_dim + j]; // load the matrix form sklearn.decomposition
                }
            }
        }

        Eigen::MatrixXf X_;
        Eigen::MatrixXd vec, val;
        unsigned nd_,dimension_,proj_dim;
        float *mean_;
    };

}



#endif //LEARN_TO_PRUNE_PCA_H
