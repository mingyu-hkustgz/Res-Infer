cd ..
source set.sh

pca_recall=0.995
# Learn Linear Model Without Negative Sample

for data in "${datasets[@]}"; do
  echo "precompute - ${data}"
  for K in {20,100}; do
    data_path=${store_path}/${data}
    index_path=./DATA/${data}
    base="${index_path}/${data}_base_pca.fvecs"
    learn="${data_path}/${data}_learn.fvecs"
    ground="${data_path}/${data}_learn_groundtruth.ivecs"
    trans="${index_path}/${data}_pca_matrix.fvecs"
    linear="${index_path}/linear/linear_${K}_l2.log"
    ./cmake-build-debug/src/binery_search_parameter -d 0 -n $base -q $learn -g $ground -t $trans -l $linear -k $K -e $pca_recall

    linear="${index_path}/linear/linear_${K}_ip.log"
    ./cmake-build-debug/src/binery_search_parameter -d 1 -n $base -q $learn -g $ground -t $trans -l $linear -k $K -e $pca_recall
  done
done

# Learn Linear Model With Negative Sample For HNSW
#for K in {20,100}; do
#  for data in "${datasets[@]}"; do
#    echo "Indexing - ${data}"
#    if [ $data == "_tiny5m" ]; then
#      efSearch=1000
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "_msong" ]; then
#      efSearch=200
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "_word2vec" ]; then
#      efSearch=2000
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "_glove2.2m" ]; then
#      efSearch=1000
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "gist" ]; then
#      efSearch=500
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "deep1M" ]; then
#      efSearch=500
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "sift" ]; then
#      efSearch=200
#      opq_recall=0.995
#      pca_recall=0.995
#    fi
#
#    data_path=${store_path}/${data}
#    pre_data=./DATA/${data}
#
#    index="${pre_data}/${data}_ef500_M16_opq.index"
#    learn="${data_path}/${data}_learn.fvecs"
#    ground="${data_path}/${data}_learn_groundtruth.ivecs"
#    trans="${pre_data}/${data}_opq_matrix.fvecs"
#    code_book="${pre_data}/${data}_codebook.centroid"
#
#    index_type="hnsw1"
#    linear="${pre_data}/linear/linear_${index_type}_opq_${K}.log"
#    logger="./logger/${data}_logger_opq_${index_type}.fvecs"
#
#    ./cmake-build-debug/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
#
#    python ./data/linear.py -d ${data} -m "opq" -i ${index_type} -k ${K}
#
#    ./cmake-build-debug/src/logger_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
#
#    trans="${pre_data}/${data}_pca_matrix.fvecs"
#    index="${pre_data}/${data}_ef500_M16_pca.index"
#    linear="${pre_data}/linear/linear_${index_type}_pca_${K}.log"
#    logger="./logger/${data}_logger_pca_${index_type}.fvecs"
#
#    ./cmake-build-debug/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
#
#    python ./data/linear.py -d ${data} -m "pca" -i ${index_type} -k ${K}
#
#    ./cmake-build-debug/src/logger_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
#
#  done
#
#done

# Learn Linear Model With Negative Sample For IVF

#for K in {20,100}; do
#  for data in "${datasets[@]}"; do
#    echo "Indexing - ${data}"
#
#    if [ $data == "_tiny5m" ]; then
#      efSearch=200
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "_msong" ]; then
#      efSearch=100
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "_word2vec" ]; then
#      efSearch=100
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "_glove2.2m" ]; then
#      efSearch=200
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "gist" ]; then
#      efSearch=100
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "deep1M" ]; then
#      efSearch=100
#      opq_recall=0.995
#      pca_recall=0.995
#    elif [ $data == "sift" ]; then
#      efSearch=100
#      opq_recall=0.995
#      pca_recall=0.995
#    fi
#
#    data_path=${store_path}/${data}
#    index_path=./DATA/${data}
#
#    index="${index_path}/${data}_ivf_opq.index"
#    linear="${index_path}/linear/linear_ivf_opq_${K}.log"
#    learn="${data_path}/${data}_learn.fvecs"
#    ground="${data_path}/${data}_learn_groundtruth.ivecs"
#    trans="${index_path}/${data}_opq_matrix.fvecs"
#    code_book="${index_path}/${data}_codebook.centroid"
#    logger="./logger/${data}_logger_opq_ivf.fvecs"
#
#    ./cmake-build-debug/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
#
#    python ./data/linear.py -d ${data} -m "opq" -i "ivf" -k ${K}
#
#    ./cmake-build-debug/src/logger_ivf_opq -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall}
#
#    index="${index_path}/${data}_ivf2_pca.index"
#    linear="${index_path}/linear/linear_ivf_pca_${K}.log"
#    trans="${index_path}/${data}_pca_matrix.fvecs"
#    logger="./logger/${data}_logger_pca_ivf.fvecs"
#
#    ./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
#
#    python ./data/linear.py -d ${data} -m "pca" -i "ivf" -k ${K}
#
#    ./cmake-build-debug/src/logger_ivf_pca -d 2 -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -o ${logger} -k ${K} -s ${efSearch} -e ${pca_recall}
#
#  done
#
#done