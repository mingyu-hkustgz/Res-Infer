

cd ..

data='deep1M'
data_path=/home/DATA/vector_data/${data}
index_path=/home/DATA/graph_data/hnsw
result_path=./results/
temp_data=./DATA/${data}
ef=500
M=16
K=100

for randomize in {0..2}
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

#./cmake-build-debug/src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans}

done

proj_dim=64

index="${index_path}/${data}_ef500_M16_opq.index"
trans="${temp_data}/${data}_opq_matrix_${proj_dim}.fvecs"
code_book="${temp_data}/${data}_codebook_${proj_dim}.centroid"

for randomize in {3..4}
do

res="${result_path}/${data}_ad_hnsw_${randomize}.log"
linear="${temp_data}/linear_hnsw0_opq_${proj_dim}.log"

# ./cmake-build-debug/src/search_hnsw_opq -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear}

done

for randomize in {5..6}
do

res="${result_path}/${data}_ad_hnsw_${randomize}.log"
linear="${temp_data}/linear_hnsw1_opq_${proj_dim}.log"

# ./cmake-build-debug/src/search_hnsw_opq -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear}

done

proj_dim=32
method="0"
index="${index_path}/${data}_ef500_M16_pca.index"
trans="${temp_data}/${data}_pca_matrix_${proj_dim}.fvecs"
for randomize in {7..8}
do

linear="${temp_data}/linear_hnsw${method}_pca_${proj_dim}.log"
res="${result_path}/${data}_ad_hnsw_${randomize}.log"

./cmake-build-debug/src/search_hnsw_pca -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear}
method="1"
done

