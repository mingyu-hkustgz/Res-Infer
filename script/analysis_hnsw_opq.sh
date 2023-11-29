cd ..
source set.sh

efConstruction=500
M=16
nbits=8
pca_dim=32
data="gist"

for opq_dim in 240;
do

python ./data/opq.py -d ${data} -m ${opq_dim} -b ${nbits}

data_path=${store_path}/${data}
index_path=./DATA/${data}
pre_path=./DATA/${data}

data_file="${pre_path}/${data}_base_opq_${opq_dim}.fvecs"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq_${opq_dim}.index"
./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M

done