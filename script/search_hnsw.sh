
efSearch=50
sigma=8
cd ..

source set.sh

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "_tiny5m" ]; then
      efSearch=500
      sigma=12
    elif [ $data == "_msong" ]; then
      efSearch=30
      sigma=12
    elif [ $data == "_word2vec" ]; then
      efSearch=500
      sigma=12
    elif [ $data == "_glove2.2m" ]; then
      efSearch=500
      sigma=16
    elif [ $data == "gist" ]; then
      efSearch=250
      sigma=10
    elif [ $data == "deep1M" ]; then
      efSearch=100
      sigma=8
    elif [ $data == "_sift10m" ]; then
      efSearch=50
      sigma=8
    elif [ $data == "deep100M" ]; then
      efSearch=500
      sigma=8
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path="./results/recall@${K}/${data}"
    temp_data=./DATA/${data}
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    ef=500
    M=16

    for randomize in {0..1}; do
      if [ $randomize == "1" ]; then
        echo "HNSW++"
        index="${index_path}/O${data}_ef${ef}_M${M}.index"
      elif [ $randomize == "2" ]; then
        echo "HNSW+"
        index="${index_path}/O${data}_ef${ef}_M${M}.index"
      else
        echo "HNSW"
        index="${index_path}/${data}_ef${ef}_M${M}.index"
      fi

      res="${result_path}/${data}_ad_hnsw_${randomize}.log"
      trans="${temp_data}/O.fvecs"
      #./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -s ${efSearch} &
    done


    index="${index_path}/${data}_ef500_M16_pca.index"
    trans="${temp_data}/${data}_pca_matrix.fvecs"
    randomize=6
    res="${result_path}/${data}_ad_hnsw_5.log"
    linear="${index_path}/linear/linear_${K}_l2.log"
    #./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} &

    index="${index_path}/${data}_ef500_M16_pca.index"
    trans="${temp_data}/${data}_pca_matrix.fvecs"
    randomize=6
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    linear="${index_path}/linear/linear_hnsw1_pca_${K}.log"
    #./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch} &

    index="${index_path}/${data}_ef500_M16_pca.index"
    trans="${temp_data}/${data}_pca_matrix_${K}.fvecs"
    randomize=7
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    ./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch} &

    index="${index_path}/${data}_ef500_M16_pca.index"
    trans="${temp_data}/${data}_pca_matrix_${K}.fvecs"
    randomize=8
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    ./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -e ${sigma} -k ${K} -s ${efSearch} &

    wait

    index="${index_path}/${data}_ef500_M16_opq.index"
    trans="${temp_data}/${data}_opq_matrix.fvecs"
    randomize=3
    code="${temp_data}/${data}_codebook.centroid"
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    linear="${index_path}/linear/linear_hnsw1_opq_${K}.log"
    #./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}

    index="${index_path}/${data}_ef500_M16_opq.index"
    trans="${temp_data}/${data}_opq_matrix.fvecs"
    randomize=4
    code="${temp_data}/${data}_codebook.centroid"
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    linear="${index_path}/linear/linear_hnsw1_opq_${K}.log"
    #./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -b ${code} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}

  done
  wait
done
