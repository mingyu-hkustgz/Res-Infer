
cd ..


efConstruction=500
M=16
data='gist'


echo "Indexing - ${data}"

data_path=/home/BLD/mingyu/DATA/vector_data/${data}
index_path=/home/BLD/mingyu/DATA/graph_data/hnsw

data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

data_file="${data_path}/O${data}_base.fvecs"
index_file="${index_path}/O${data}_ef${efConstruction}_M${M}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
