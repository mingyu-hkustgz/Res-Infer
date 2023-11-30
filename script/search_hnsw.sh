opq_dim=32
pca_dim=32
efSearch=50

cd ..

source set.sh

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "_tiny5m" ]; then
      opq_dim=96
      efSearch=500
    elif [ $data == "_msong" ]; then
      opq_dim=105
      efSearch=30
    elif [ $data == "_word2vec" ]; then
      opq_dim=75
      efSearch=500
    elif [ $data == "_glove2.2m" ]; then
      opq_dim=75
      efSearch=500
    elif [ $data == "gist" ]; then
      opq_dim=120
      efSearch=250
    elif [ $data == "deep1M" ]; then
      opq_dim=64
      efSearch=100
    elif [ $data == "_sift10m" ]; then
      opq_dim=32
      efSearch=50
    elif [ $data == "deep100M" ]; then
      opq_dim=24
      efSearch=500
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
#      ./cmake-build-debug/src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -s ${efSearch} &
    done

    index="${index_path}/${data}_ef500_M16_opq.index"
    trans="${temp_data}/${data}_opq_matrix_${opq_dim}.fvecs"
    code_book="${temp_data}/${data}_codebook_${opq_dim}.centroid"

    randomize=5

    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    linear="${temp_data}/linear/linear_hnsw1_opq_${opq_dim}_${K}.log"

    ./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear} -s ${efSearch}

    randomize=6

    res="${result_path}/${data}_ad_hnsw_${randomize}.log"
    linear="${temp_data}/linear/linear_hnsw1_opq_${opq_dim}_${K}.log"

    ./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear} -s ${efSearch}

    index="${index_path}/${data}_ef500_M16_pca.index"
    trans="${temp_data}/${data}_pca_matrix_${pca_dim}.fvecs"
    randomize=8
    linear="${temp_data}/linear/linear_hnsw1_pca_${pca_dim}_${K}.log"
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"

#    ./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear} -s ${efSearch} &

  done
  wait
done
