cd ..

data='deep1M'
echo "Indexing - ${data}"

data_path=/home/DATA/vector_data/${data}
pre_data=./DATA/${data}
K=100
PCA_dim=32
OPQ_dim=64

index="/home/DATA/graph_data/hnsw/${data}_ef500_M16_opq.index"
index_type="hnsw0"
linear="${pre_data}/linear_${index_type}_opq_${OPQ_dim}.log"
learn="${data_path}/${data}_learn.fvecs"
trans="${pre_data}/${data}_opq_matrix_${OPQ_dim}.fvecs"
code_book="${pre_data}/${data}_codebook_${OPQ_dim}.centroid"
logger="./logger/${data}_logger_opq_${OPQ_dim}_${index_type}.fvecs"

./cmake-build-debug/src/logger_hnsw_opq -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k 100

python ./data/linear.py -d ${data} -m "opq" -p ${OPQ_dim} -i ${index_type}

./cmake-build-debug/src/logger_hnsw_opq -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k 100

index_type="hnsw1"
linear="${pre_data}/linear_${index_type}_opq_${OPQ_dim}.log"
logger="./logger/${data}_logger_opq_${OPQ_dim}_${index_type}.fvecs"

./cmake-build-debug/src/logger_hnsw_opq -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k 100

python ./data/linear.py -d ${data} -m "opq" -p ${OPQ_dim} -i ${index_type}

./cmake-build-debug/src/logger_hnsw_opq -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k 100


index_type="hnsw0"
linear="${pre_data}/linear_${index_type}_pca_${PCA_dim}.log"
trans="${pre_data}/${data}_pca_matrix_${PCA_dim}.fvecs"
logger="./logger/${data}_logger_pca_${PCA_dim}_${index_type}.fvecs"
index="/home/DATA/graph_data/hnsw/${data}_ef500_M16_pca.index"

./cmake-build-debug/src/logger_hnsw_pca -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k 100

python ./data/linear.py -d ${data} -m "pca" -p ${PCA_dim} -i "${index_type}"

./cmake-build-debug/src/logger_hnsw_pca -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k 100

index_type="hnsw1"
linear="${pre_data}/linear_${index_type}_pca_${PCA_dim}.log"
logger="./logger/${data}_logger_pca_${PCA_dim}_${index_type}.fvecs"

./cmake-build-debug/src/logger_hnsw_pca -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k 100

python ./data/linear.py -d ${data} -m "pca" -p ${PCA_dim} -i ${index_type}

./cmake-build-debug/src/logger_hnsw_pca -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k 100

