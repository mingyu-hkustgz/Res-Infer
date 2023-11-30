cd ..
source set.sh

efConstruction=500
M=16

pca_dim=32
opq_dim=96

for data in "${datasets[@]}"; do
  echo "Indexing - ${data}"

  if [ $data == "_tiny5m" ]; then
    opq_dim=96
  elif [ $data == "_msong" ]; then
    opq_dim=105
  elif [ $data == "_word2vec" ]; then
    opq_dim=75
  elif [ $data == "_glove2.2m" ]; then
    opq_dim=75
  elif [ $data == "gist" ]; then
    opq_dim=120
  elif [ $data == "deep1M" ]; then
    opq_dim=64
  elif [ $data == "_sift10m" ]; then
    opq_dim=32
  elif [ $data == "deep100M" ]; then
    opq_dim=24
  fi

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${pre_path}/O${data}_base.fvecs"
index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${pre_path}/${data}_base_pca_${pca_dim}.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}_pca.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${pre_path}/${data}_base_opq_${opq_dim}.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

done
