cd ..
source set.sh

efConstruction=500
M=16

for data in "${datasets[@]}"; do

  echo "precompute - ${data}"

  python ./data/pca.py -d ${data}

  echo "Indexing - ${data}"

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  data_file="${pre_path}/${data}_base_pca.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_pca.index"
  ./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

done


for data in "${datasets[@]}"; do

  echo "Indexing - ${data}"

  python ./data/ivf.py -d ${data} -m "pca"
  data_file="${index_path}/${data}_base_pca.fvecs"
  centroid_file="${index_path}/${data}_centroid_pca.fvecs"
  index_file="${index_path}/${data}_ivf2_pca.index"
  adaptive=2
  ./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

  data_file="${index_path}/${data}_base_pca.fvecs"
  centroid_file="${index_path}/${data}_centroid_pca.fvecs"
  index_file="${index_path}/${data}_ivf1_pca.index"
  adaptive=1
  ./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

done

