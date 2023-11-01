cd ..
source set.sh
pca_dim=32
opq_dim=32
efSearch=50

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Indexing - ${data}"
    if [ $data == "_tiny5m" ]; then
      opq_dim=96
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "_msong" ]; then
      opq_dim=105
      efSearch=200
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "_word2vec" ]; then
      opq_dim=75
      efSearch=2000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "_glove2.2m" ]; then
      opq_dim=75
      efSearch=1000
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "gist" ]; then
      opq_dim=120
      efSearch=500
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "deep1M" ]; then
      opq_dim=64
      efSearch=500
      opq_recall=0.995
      pca_recall=0.995
    elif [ $data == "sift" ]; then
      opq_dim=32
      efSearch=200
      opq_recall=0.995
      pca_recall=0.995
    fi

    data_path=${store_path}/${data}
    pre_data=./DATA/${data}

    index="/home/DATA/graph_data/hnsw/${data}_ef500_M16_opq.index"
    learn="${data_path}/${data}_learn.fvecs"
    ground="${data_path}/${data}_learn_groundtruth.ivecs"
    trans="${pre_data}/${data}_opq_matrix_${opq_dim}.fvecs"
    code_book="${pre_data}/${data}_codebook_${opq_dim}.centroid"

    index_type="hnsw1"
    linear="${pre_data}/linear/linear_${index_type}_opq_${opq_dim}_${K}.log"
    logger="./logger/${data}_logger_opq_${opq_dim}_${index_type}.fvecs"

    ./cmake-build-debug/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}

    python ./data/linear.py -d ${data} -m "opq" -p ${opq_dim} -i ${index_type} -k ${K}

    ./cmake-build-debug/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}

    trans="${pre_data}/${data}_pca_matrix_${pca_dim}.fvecs"
    index="/home/DATA/graph_data/hnsw/${data}_ef500_M16_pca.index"
    linear="${pre_data}/linear/linear_${index_type}_pca_${pca_dim}_${K}.log"
    logger="./logger/${data}_logger_pca_${pca_dim}_${index_type}.fvecs"

    ./cmake-build-debug/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

    python ./data/linear.py -d ${data} -m "pca" -p ${pca_dim} -i ${index_type} -k ${K}

    ./cmake-build-debug/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

  done

done
