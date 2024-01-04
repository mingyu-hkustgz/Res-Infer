cd ..
source set.sh
pca_dim=32
efSearch=50

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Indexing - ${data}"
    if [ $data == "_tiny5m" ]; then
      efSearch=1000
      pca_recall=0.995
    elif [ $data == "_msong" ]; then
      efSearch=200
      pca_recall=0.995
    elif [ $data == "_word2vec" ]; then
      efSearch=2000
      pca_recall=0.995
    elif [ $data == "_glove2.2m" ]; then
      efSearch=1000
      pca_recall=0.995
    elif [ $data == "gist" ]; then
      efSearch=500
      pca_recall=0.995
    elif [ $data == "deep1M" ]; then
      efSearch=500
      pca_recall=0.995
    elif [ $data == "_sift10m" ]; then
      efSearch=500
      pca_recall=0.995
    elif [ $data == "deep100M" ]; then
      efSearch=500
      pca_recall=0.995
    fi

    data_path=${store_path}/${data}
    pre_data=./DATA/${data}
    learn="${data_path}/${data}_learn.fvecs"
    ground="${data_path}/${data}_learn_groundtruth.ivecs"

    trans="${pre_data}/${data}_pca_matrix_${pca_dim}.fvecs"
    index="${pre_data}/${data}_ef500_M16_pca.index"
    linear="${pre_data}/linear/linear_${index_type}_pca_${pca_dim}_${K}.log"
    logger="./logger/${data}_logger_pca_${pca_dim}_${index_type}.fvecs"

    ./cmake-build-debug/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

    python ./data/linear.py -d ${data} -m "pca" -p ${pca_dim} -i ${index_type} -k ${K}

    ./cmake-build-debug/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

  done

done
