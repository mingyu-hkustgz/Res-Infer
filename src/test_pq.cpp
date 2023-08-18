//
// Created by mingyu on 23-7-20.
//
#include "utils.h"
#include "pq.h"
#include "ivf.h"
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
    Index_PQ::Quantizer PQ(points_num, dim);
    PQ.load_product_codebook(argv[4]);
    PQ.load_project_matrix(argv[5]);
    IVF ivf;
    ivf.load(argv[6]);
    ivf.PQ = &PQ;
    ivf.L1_data = data_load;
    ivf.encoder_origin_data();
    std::cout << "IVF naive centroid distance" << std::endl;

    ivf.PQ->project_vector(test_load, test_num);

    ivf.PQ->calc_dist_map(test_load);
    double app = ivf.PQ->naive_product_map_dist(19053);
    double acc = naive_l2_dist_calc(ivf.L1_data + 19053 * ivf.D, test_load, ivf.D);
    double cluster = ivf.PQ->node_cluster_dist_[19053];
    std::cout<<app<<" "<<acc<<" "<<cluster<<endl;


    return 0;
}