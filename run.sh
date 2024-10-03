bash make_dir.sh

cd ./script

bash pre_compute.sh

bash index_ivf.sh

bash index_hnsw.sh

bash index_pca.sh

bash index_opq.sh

bash linear.sh

bash search_ivf.sh

bash search_hnsw.sh

#bash search_hnsw_finger.sh

#bash search_hnsw_avx512.sh

#bash search_ivf_avx512.sh