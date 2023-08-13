cd ..

data='deep1M'
data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}
hnsw_path=/home/DATA/graph_data/hnsw
result_path=./results/

ef=500
M=16
K=100
proj_dim=32
method="0"

for randomize in {7..8}
do

res="${result_path}/${data}_ad_hnsw_${randomize}.log"
index="${hnsw_path}/${data}_ef500_M16_pca.index"
query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_pca_matrix_${proj_dim}.fvecs"
linear="${index_path}/linear_hnsw${method}_pca_${proj_dim}.log"
./cmake-build-debug/src/search_hnsw_pca -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear}
method="1"
done


