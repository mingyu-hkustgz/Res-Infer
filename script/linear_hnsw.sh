cd ..

K=100
pca_dim=32
opq_dim=32
efSearch=50
for data in {gist,deep1M,_tiny5m,_word2vec,_msong,_glove2.2m}
do
echo "Indexing - ${data}"

if [ $data == "_tiny5m" ]
then
    opq_dim=96
    efSearch=1000
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "_msong" ]
then
    opq_dim=105
    efSearch=100
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "_word2vec" ]
then
    opq_dim=75
    efSearch=1000
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "_glove2.2m" ]
then
    opq_dim=75
    efSearch=1000
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "gist" ]
then
    opq_dim=120
    efSearch=500
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "deep1M" ]
then
    opq_dim=64
    efSearch=500
    opq_eps=0.99999
    pca_eps=0.999999
fi


data_path=/home/DATA/vector_data/${data}
pre_data=./DATA/${data}

index="/home/DATA/graph_data/hnsw/${data}_ef500_M16_opq.index"
index_type="hnsw0"
linear="${pre_data}/linear_${index_type}_opq_${opq_dim}.log"
learn="${data_path}/${data}_learn.fvecs"
trans="${pre_data}/${data}_opq_matrix_${opq_dim}.fvecs"
code_book="${pre_data}/${data}_codebook_${opq_dim}.centroid"
logger="./logger/${data}_logger_opq_${opq_dim}_${index_type}.fvecs"

./cmake-build-debug/src/logger_hnsw_opq -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_eps}

python ./data/linear.py -d ${data} -m "opq" -p ${opq_dim} -i ${index_type}

./cmake-build-debug/src/logger_hnsw_opq -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_eps}

index_type="hnsw1"
linear="${pre_data}/linear_${index_type}_opq_${opq_dim}.log"
logger="./logger/${data}_logger_opq_${opq_dim}_${index_type}.fvecs"

./cmake-build-debug/src/logger_hnsw_opq -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_eps}

python ./data/linear.py -d ${data} -m "opq" -p ${opq_dim} -i ${index_type}

./cmake-build-debug/src/logger_hnsw_opq -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_eps}


index_type="hnsw0"
linear="${pre_data}/linear_${index_type}_pca_${pca_dim}.log"
trans="${pre_data}/${data}_pca_matrix_${pca_dim}.fvecs"
logger="./logger/${data}_logger_pca_${pca_dim}_${index_type}.fvecs"
index="/home/DATA/graph_data/hnsw/${data}_ef500_M16_pca.index"

./cmake-build-debug/src/logger_hnsw_pca -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_eps}

python ./data/linear.py -d ${data} -m "pca" -p ${pca_dim} -i "${index_type}"

./cmake-build-debug/src/logger_hnsw_pca -d 0 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_eps}

index_type="hnsw1"
linear="${pre_data}/linear_${index_type}_pca_${pca_dim}.log"
logger="./logger/${data}_logger_pca_${pca_dim}_${index_type}.fvecs"

./cmake-build-debug/src/logger_hnsw_pca -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_eps}

python ./data/linear.py -d ${data} -m "pca" -p ${pca_dim} -i ${index_type}

./cmake-build-debug/src/logger_hnsw_pca -d 1 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_eps}

done