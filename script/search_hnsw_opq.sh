cd ..

data='deep1M'
data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}
hnsw_path=/home/DATA/graph_data/hnsw
result_path=./results/

ef=500
M=16
K=100
proj_dim=64

for randomize in {3..6}
do

res="${result_path}/${data}_ad_hnsw_${randomize}.log"
index="${hnsw_path}/${data}_ef500_M16_opq.index"
query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_opq_matrix_${proj_dim}.fvecs"
code_book="${index_path}/${data}_codebook_${proj_dim}.centroid"

./cmake-build-debug/src/search_hnsw_opq -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book}

done


