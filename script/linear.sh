cd ..
source set.sh

pca_recall=0.995

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

