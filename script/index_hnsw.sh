cd ..
source set.sh

efConstruction=500
M=16

pca_dim=32

for data in "${datasets[@]}"; do
  echo "Indexing - ${data}"

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${pre_path}/O${data}_base.fvecs"
index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

done
