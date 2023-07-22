//
// Created by mingyu on 23-7-22.
//
#include "utils.h"
#include "pca.h"
#include <iostream>
using namespace std;

int main(int argc, char **argv) {
    float *data_load = nullptr;
    unsigned points_num, dim;
    float *test_load = nullptr;
    unsigned test_num, test_dim;
    float *train_load = nullptr;
    unsigned train_num, train_dim;
    cout << "data loading..." << endl;
    load_float_data(argv[1], data_load, points_num, dim);
    load_float_data(argv[2], train_load, train_num, train_dim);
    load_float_data(argv[3], test_load, test_num, test_dim);
    Index_PCA::PCA PCA(points_num, dim, data_load);
    PCA.load_project_matrix(argv[4]);
    PCA.load_pca_mean_(argv[5]);
    std::cout<<"pca project vector"<<std::endl;
    for(int i=0;i<3;i++){
        for(int j=0;j<dim;j++){
            std::cout<<train_load[i *dim + j]<<" ";
        }
        std::cout<<endl;
    }
    std::cout<<"test project vector"<<std::endl;
    PCA.pca_transform(test_load, test_num,test_dim);
    for(int i =0;i<3;i++){
        for(int j=0;j<dim;j++){
            std::cout<<test_load[i*dim + j]<<" ";
        }
        std::cout<<endl;
    }
    return 0;
}