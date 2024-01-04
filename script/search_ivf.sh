# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0

opq_dim=32
pca_dim=32
efSearch=50
C=4096
cd ..

source set.sh

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "_tiny5m" ]; then
      efSearch=100
    elif [ $data == "_msong" ]; then
      efSearch=25
    elif [ $data == "_word2vec" ]; then
      efSearch=20
    elif [ $data == "_glove2.2m" ]; then
      efSearch=100
    elif [ $data == "gist" ]; then
      efSearch=50
    elif [ $data == "deep1M" ]; then
      efSearch=50
    elif [ $data == "_sift10m" ]; then
      efSearch=30
    fi

    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    result_path="./results/recall@${K}/${data}"

    for randomize in {0..1}; do
      if [ $randomize == "1" ]; then
        echo "IVF++"
      elif [ $randomize == "2" ]; then
        echo "IVF+"

      else
        echo "IVF"
      fi

      res="${result_path}/${data}_ad_ivf_${randomize}.log"
      index="${index_path}/${data}_ivf_${C}_${randomize}.index"

      query="${data_path}/${data}_query.fvecs"
      gnd="${data_path}/${data}_groundtruth.ivecs"
      trans="${index_path}/O.fvecs"

      ./cmake-build-debug/src/search_ivf -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -s ${efSearch} &

    done

    randomize=5
    res="${result_path}/${data}_ad_ivf_${randomize}.log"
    index="${index_path}/${data}_ivf1_pca.index"

    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    trans="${index_path}/${data}_pca_matrix.fvecs"
    square="${index_path}/${data}_pca_square.fvecs"
    ./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -v ${square} -s ${efSearch} &

    randomize=6
    res="${result_path}/${data}_ad_ivf_${randomize}.log"
    index="${index_path}/${data}_ivf2_pca.index"

    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    trans="${index_path}/${data}_pca_matrix.fvecs"
    square="${index_path}/${data}_pca_square.fvecs"
    ./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -v ${square} -s ${efSearch} &
  done
  wait
done
