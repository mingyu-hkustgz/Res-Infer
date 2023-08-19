cd ..

K=100
pca_dim=32
opq_dim=32
efSearch=50
for data in _tiny5m
do
echo "Indexing - ${data}"

if [ $data == "_tiny5m" ]
then
    opq_dim=96
    efSearch=200
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "_msong" ]
then
    opq_dim=105
    efSearch=50
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "_word2vec" ]
then
    opq_dim=75
    efSearch=50
    opq_eps=0.999999
    pca_eps=0.9999999
elif [ $data == "_glove2.2m" ]
then
    opq_dim=75
    efSearch=200
    opq_eps=0.999999
    pca_eps=0.9999999
elif [ $data == "gist" ]
then
    opq_dim=120
    efSearch=200
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "deep1M" ]
then
    opq_dim=64
    efSearch=100
    opq_eps=0.99999
    pca_eps=0.999999
elif [ $data == "sift" ]
then
    opq_dim=32
    efSearch=50
    opq_eps=0.9999
    pca_eps=0.99999
fi


data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}

result_path=./results

index="${index_path}/${data}_ivf_opq_${opq_dim}.index"
linear="${index_path}/linear_ivf_opq_${opq_dim}.log"
learn="${data_path}/${data}_learn.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_opq_matrix_${opq_dim}.fvecs"
code_book="${index_path}/${data}_codebook_${opq_dim}.centroid"
logger="./logger/${data}_logger_opq_${opq_dim}_ivf.fvecs"

./cmake-build-debug/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_eps}

python ./data/linear.py -d ${data} -m "opq" -p ${opq_dim} -i "ivf"

./cmake-build-debug/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_eps}



index="${index_path}/${data}_ivf2_pca_${pca_dim}.index"
linear="${index_path}/linear_ivf_pca_${pca_dim}.log"
trans="${index_path}/${data}_pca_matrix_${pca_dim}.fvecs"
logger="./logger/${data}_logger_pca_${pca_dim}_ivf.fvecs"

./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_eps}

python ./data/linear.py -d ${data} -m "pca" -p ${pca_dim} -i "ivf"

./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_eps}

done