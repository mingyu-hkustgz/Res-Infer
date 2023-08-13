
randomize=0

cd ..
data='deep1M'
data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}
result_path=./results
C=4096
K=100
proj_dim=64

for randomize in {3..4}
do

res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf_opq_${proj_dim}.index"
linear="${index_path}/linear_ivf_opq_${proj_dim}.log"
query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_opq_matrix_${proj_dim}.fvecs"
code_book="${index_path}/${data}_codebook_${proj_dim}.centroid"

./cmake-build-debug/src/search_ivf_opq -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear}

done
