//
// Created by BLD on 23-7-20.
//
#include "utils.h"
#include "pq.h"
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
    Index_PQ::Quantizer PQ(points_num, dim, data_load);
    PQ.load_product_codebook(argv[4]);
    PQ.encoder_origin_data();
    std::cout<<endl;
    for(int i=0;i<100;i++) std::cout<<naive_l2_dist_calc(data_load + i *dim, train_load, dim)<<" ";
    std::cout<<endl;

    std::cout<<endl;
    for(int i=0;i<100;i++) std::cout<<naive_l2_dist_calc(data_load + i *dim, test_load, dim)<<" ";
    std::cout<<endl;

    PQ.load_project_matrix(argv[5]);
    std::cout<<test_num<<std::endl;
    PQ.project_vector(test_load,1);
    std::cout<<endl;
    for(int i=0;i<100;i++) std::cout<<naive_l2_dist_calc(data_load + i *dim, test_load, dim)<<" ";
    std::cout<<endl;


    return 0;
}
