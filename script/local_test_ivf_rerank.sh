# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0
sigma=8
efSearch=50
C=4096
cd ..

source set.sh

for K in {1,10}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "_tiny5m" ]; then
      efSearch=100
      sigma=12
    elif [ $data == "_msong" ]; then
      efSearch=25
      sigma=12
    elif [ $data == "_word2vec" ]; then
      efSearch=20
      sigma=12
    elif [ $data == "_glove2.2m" ]; then
      efSearch=100
      sigma=16
    elif [ $data == "gist" ]; then
      efSearch=50
      sigma=10
    elif [ $data == "deep1M" ]; then
      efSearch=50
      sigma=8
    elif [ $data == "_sift10m" ]; then
      efSearch=30
      sigma=8
    elif [ $data == "deep100M" ]; then
      efSearch=100
      sigma=12
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path="./results/recall@${K}/${data}"


    randomize=6
    res="${result_path}/${data}_ad_ivf_${randomize}.log"
    index="${index_path}/${data}_ivf_opq.index"
    code="${index_path}/${data}_codebook.centroid"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    trans="${index_path}/${data}_opq_matrix.fvecs"
    linear="${index_path}/linear/linear_ivf_opq_${K}.log"
    ./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -b ${code} -q ${query} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}

    randomize=7
    res="${result_path}/${data}_ad_ivf_${randomize}.log"
    index="${index_path}/${data}_ivf_opq.index"
    code="${index_path}/${data}_codebook.centroid"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    trans="${index_path}/${data}_opq_matrix.fvecs"
    linear="${index_path}/linear/linear_ivf_opq_${K}.log"
    ./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -b ${code} -q ${query} -g ${gnd} -r ${res} -t ${trans} -l ${linear} -k ${K} -s ${efSearch}

  done
done
