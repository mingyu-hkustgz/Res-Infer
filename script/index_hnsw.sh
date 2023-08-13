
cd ..


efConstruction=500
M=16
data='deep1M'
pca_dim=32
opq_dim=64

echo "Indexing - ${data}"

data_path=/home/BLD/mingyu/DATA/vector_data/${data}
index_path=/home/BLD/mingyu/DATA/graph_data/hnsw
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
