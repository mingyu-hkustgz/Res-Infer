

opq_dim=32
pca_dim=32
efSearch=50
K=100
cd ..
for data in gist
do
echo "Searching - ${data}"

if [ $data == "_tiny5m" ]
then
    opq_dim=96
    efSearch=500
elif [ $data == "_msong" ]
then
    opq_dim=105
    efSearch=30
elif [ $data == "_word2vec" ]
then
    opq_dim=75
    efSearch=500
elif [ $data == "_glove2.2m" ]
then
    opq_dim=75
    efSearch=500
elif [ $data == "gist" ]
then
    opq_dim=120
    efSearch=250
elif [ $data == "deep1M" ]
then
    opq_dim=64
    efSearch=100
elif [ $data == "sift" ]
then
    opq_dim=32
    efSearch=50
fi

data_path=/home/DATA/vector_data/${data}
index_path=/home/DATA/graph_data/hnsw
result_path="./results/recall@${K}/${data}"
temp_data=./DATA/${data}
ef=500
M=16

for randomize in {1..2}
do
if [ $randomize == "1" ]
then 
    echo "HNSW++"
    index="${index_path}/O${data}_ef${ef}_M${M}.index"
elif [ $randomize == "2" ]
then 
    echo "HNSW+"
    index="${index_path}/O${data}_ef${ef}_M${M}.index"
else
    echo "HNSW"
    index="${index_path}/${data}_ef${ef}_M${M}.index"
fi

res="${result_path}/${data}_ad_hnsw_${randomize}.log"
query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${temp_data}/O.fvecs"
./cmake-build-debug/src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -s ${efSearch}

done


index="${index_path}/${data}_ef500_M16_opq.index"
trans="${temp_data}/${data}_opq_matrix_${opq_dim}.fvecs"
code_book="${temp_data}/${data}_codebook_${opq_dim}.centroid"


for randomize in {5..6}
do

res="${result_path}/${data}_ad_hnsw_${randomize}.log"
linear="${temp_data}/linear_hnsw1_opq_${opq_dim}_${K}.log"

./cmake-build-debug/src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear} -s ${efSearch}

done

method="1"
randomize=8
linear="${temp_data}/linear_hnsw${method}_pca_${pca_dim}_${K}.log"
res="${result_path}/${data}_ad_hnsw_${randomize}.log"

./cmake-build-debug/src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear} -s ${efSearch}


done