cd ..

C=4096
data='deep1M'

echo "Indexing - ${data}"

data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}

result_path=./results
C=4096
K=100
PCA_dim=32
OPQ_dim=64

index="${index_path}/${data}_ivf_opq_${OPQ_dim}.index"
linear="${index_path}/linear_ivf_opq_${OPQ_dim}.log"
query="${data_path}/${data}_query.fvecs"
learn="${data_path}/${data}_learn.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_opq_matrix_${OPQ_dim}.fvecs"
code_book="${index_path}/${data}_codebook_${OPQ_dim}.centroid"
logger="./logger/${data}_logger_opq_${OPQ_dim}_ivf.fvecs"

#./cmake-build-debug/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k 100
#
#python ./data/linear.py -d ${data} -m "opq" -p ${OPQ_dim} -i "ivf"
#
#./cmake-build-debug/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k 100



index="${index_path}/${data}_ivf2_pca_${PCA_dim}.index"
linear="${index_path}/linear_ivf_pca_${PCA_dim}.log"
trans="${index_path}/${data}_pca_matrix_${PCA_dim}.fvecs"
logger="./logger/${data}_logger_pca_${PCA_dim}_ivf.fvecs"

./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k 100

python ./data/linear.py -d ${data} -m "pca" -p ${PCA_dim} -i "ivf"

./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k 100
