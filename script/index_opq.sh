cd ..
source set.sh

efConstruction=500
M=16

for data in "${datasets[@]}"; do

  echo "precompute - ${data}"

  python ./data/opq.py -d ${data}

  echo "Indexing - HNSW - ${data}"

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  data_file="${pre_path}/${data}_base_opq.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq.index"
  ./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

  echo "Indexing -IVF- ${data}"

  python ./data/ivf.py -d ${data} -m "opq"

  data_file="${index_path}/${data}_base_opq.fvecs"
  centroid_file="${index_path}/${data}_centroid_opq.fvecs"
  index_file="${index_path}/${data}_ivf_opq.index"
  adaptive=0
  ./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive


done
