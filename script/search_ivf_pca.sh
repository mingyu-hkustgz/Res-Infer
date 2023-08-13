
randomize=5
cd ..
data='deep1M'
data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}
result_path=./results
C=4096
K=100
proj_dim=32

res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf1_pca_${proj_dim}.index"

query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_pca_matrix_${proj_dim}.fvecs"
linear="${index_path}/linear_ivf_pca_${proj_dim}.log"

./cmake-build-debug/src/search_ivf_pca -d 1 -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear}

randomize=6
res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf2_pca_${proj_dim}.index"

./cmake-build-debug/src/search_ivf_pca -d 2 -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear}

