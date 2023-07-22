//
// Created by BLD on 23-7-22.
//
#include "utils.h"

#ifndef LEARN_TO_PRUNE_PCA_H
#define LEARN_TO_PRUNE_PCA_H
namespace Index_PCA{
    class PCA{
        public:
            PCA(unsigned num, unsigned dim, float *data){
                nd_ = num;
                dimension_ = dim;// the project dim
                data_ = data;
            }



        void pca_transform(float *raw_data, unsigned num,unsigned dim) {
            Eigen::MatrixXf Q(num, dim);
            for (int i = 0; i < num; i++) {
                for (int j = 0; j < dim; j++) {
                    Q(i, j) = raw_data[i * dim + j] - mean_[j];// feature normalization
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
            unsigned origin_dim, project_dim;
            load_float_data(filename, raw_data, origin_dim, project_dim);
            X_ = Eigen::MatrixXf(origin_dim, project_dim);
            for (int i = 0; i < origin_dim; i++) {
                for (int j = 0; j < project_dim; j++) {
                    X_(i, j) = raw_data[i * project_dim + j]; // load the matrix form sklearn.decomposition
                }
            }
        }
        void load_pca_mean_(const char *filename){
            unsigned dim,num;
            load_float_data(filename, mean_, num,dim);
        }



        float *mean_;
        Eigen::MatrixXf X_;
        Eigen::MatrixXd vec, val;
        unsigned nd_,dimension_;
        float *data_;
    };

}



#endif //LEARN_TO_PRUNE_PCA_H
