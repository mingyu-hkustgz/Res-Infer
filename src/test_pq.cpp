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
    Index_PQ::Quantizer PQ(points_num, dim, data_load);
    PQ.load_product_codebook(argv[4]);
    PQ.encoder_origin_data();
    PQ.load_project_matrix(argv[5]);
    IVF ivf;
    ivf.load(argv[6]);
    ivf.PQ = &PQ;

    std::cout << "IVF naive centroid distance" << std::endl;
    std::cout << endl;
    for (int i = 0; i < 20; i++) std::cout << naive_l2_dist_calc(ivf.centroids + i * dim, test_load, dim) << " ";
    std::cout << endl;

    std::cout << endl;
    for (int i = 0; i < 20; i++) std::cout << naive_l2_dist_calc(ivf.L1_data + i * dim, test_load, dim) << " ";
    std::cout << endl;

    ivf.transform_data_opq(data_load);
    ivf.PQ->project_vector(test_load,test_num);

    std::cout << endl;
    std::cout << "IVF project centroid distance" << std::endl;
    std::cout << endl;

    for (int i = 0; i < 20; i++) std::cout << naive_l2_dist_calc(ivf.centroids + i * dim, test_load, dim) << " ";
    std::cout << endl;

    std::cout << endl;
    for (int i = 0; i < 20; i++) std::cout << naive_l2_dist_calc(ivf.L1_data + i * dim, test_load, dim) << " ";
    std::cout << endl;
    std::cout << endl;


    double res = 0.0;
    for (int i = 0; i < 40; i++) {
        ivf.PQ->calc_dist_map(test_load);
        std::cout << PQ.naive_product_map_dist(i) << " " << PQ.naive_product_dist(i,test_load)<<" ";
    }
    std::cout << endl;
    std::cout << "loss " << res / 40 << std::endl;
    std::cout << std::endl;


//    std::cout << "PQ test simd sse" << std::endl;
//    PQ.calc_dist_map(test_load);
//    unsigned NSQ = 120;
//    float sse_res[4];
//    for (int i = 0; i < 40; i += 4) {
//        __m128 candidate_dist;
//        const uint8_t *const pqcode0 = PQ.pq_mp + i * NSQ;
//        const uint8_t *const pqcode1 = pqcode0 + NSQ;
//        const uint8_t *const pqcode2 = pqcode0 + 2 * NSQ;
//        const uint8_t *const pqcode3 = pqcode0 + 3 * NSQ;
//
//        //std::cout<<(int)pqcode0[0]<<" "<<(int)pqcode1[0]<<" "<<(int)pqcode2[0]<<" "<<(int)pqcode3[0]<<endl;
//        PQ.sse4_product_map_dist(pqcode0, pqcode1, pqcode2, pqcode3, PQ.dist_mp, candidate_dist);
//        _mm_store_ps(sse_res, candidate_dist);
//        std::cout << sse_res[0] << " " << sse_res[1] << " " << sse_res[2] << " " << sse_res[3] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "PQ test simd " << std::endl;
//    PQ.calc_dist_map(test_load);
//    float avx_res[8];
//    for (int i = 0; i < 40; i += 8) {
//        __m256 candidate_dist;
//        const uint8_t *const pqcode0 = PQ.pq_mp + i * NSQ;
//        const uint8_t *const pqcode1 = pqcode0 + NSQ;
//        const uint8_t *const pqcode2 = pqcode0 + 2 * NSQ;
//        const uint8_t *const pqcode3 = pqcode0 + 3 * NSQ;
//        const uint8_t *const pqcode4 = pqcode0 + 4 * NSQ;
//        const uint8_t *const pqcode5 = pqcode0 + 5 * NSQ;
//        const uint8_t *const pqcode6 = pqcode0 + 6 * NSQ;
//        const uint8_t *const pqcode7 = pqcode0 + 7 * NSQ;
//        //std::cout<<(int)pqcode0[0]<<" "<<(int)pqcode1[0]<<" "<<(int)pqcode2[0]<<" "<<(int)pqcode3[0]<<" "<<(int)pqcode4[0]<<" "<<(int)pqcode5[0]<<" "<<(int)pqcode6[0]<<" "<<(int)pqcode7[0]<<endl;
//        PQ.axv8_product_map_dist(pqcode0, pqcode1, pqcode2, pqcode3, pqcode4, pqcode5, pqcode6, pqcode7, PQ.dist_mp,
//                                 candidate_dist);
//        _mm256_store_ps(avx_res, candidate_dist);
//        std::cout << avx_res[0] << " " << avx_res[1] << " " << avx_res[2] << " " << avx_res[3] << " " << avx_res[4]
//                  << " " << avx_res[5] << " " << avx_res[6] << " " << avx_res[7] << " ";
//    }
//    std::cout << std::endl;

    return 0;
}