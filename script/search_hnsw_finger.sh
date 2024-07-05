
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
      efSearch=200
      sigma=12
    elif [ $data == "sift100m" ]; then
      efSearch=100
      sigma=8
    elif [ $data == "_tiny80M" ]; then
      efSearch=500
      sigma=12
    fi
    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path="./results/recall@${K}/${data}"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    ef=500
    M=16
    randomize=9
    echo "HNSW"
    index="${index_path}/${data}_ef${ef}_M${M}.index"
    res="${result_path}/${data}_ad_hnsw_${randomize}.log"

    ./cmake-build-debug/src/search_hnsw -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -k ${K} -s ${efSearch} -n ${data}

  done
done
