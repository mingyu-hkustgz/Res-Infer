cd ..
source set.sh
pca_dim=32
efSearch=50

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Indexing - ${data}"

    if [ $data == "_tiny5m" ]; then
      efSearch=200
      pca_recall=0.995
    elif [ $data == "_msong" ]; then
      efSearch=100
      pca_recall=0.995
    elif [ $data == "_word2vec" ]; then
      efSearch=100
      pca_recall=0.995
    elif [ $data == "_glove2.2m" ]; then
      efSearch=200
      pca_recall=0.995
    elif [ $data == "gist" ]; then
      efSearch=100
      pca_recall=0.995
    elif [ $data == "deep1M" ]; then
      efSearch=100
      pca_recall=0.995
    elif [ $data == "_sift10m" ]; then
      efSearch=100
      pca_recall=0.995
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    learn="${data_path}/${data}_learn.fvecs"
    ground="${data_path}/${data}_learn_groundtruth.ivecs"

    index="${index_path}/${data}_ivf2_pca_${pca_dim}.index"
    linear="${index_path}/linear/linear_ivf_pca_${pca_dim}_${K}.log"
    trans="${index_path}/${data}_pca_matrix_${pca_dim}.fvecs"
    logger="./logger/${data}_logger_pca_${pca_dim}_ivf.fvecs"

    ./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

    python ./data/linear.py -d ${data} -m "pca" -p ${pca_dim} -i "ivf" -k ${K}

    ./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}

  done

done
