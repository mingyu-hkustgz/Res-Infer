
# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0

cd ..
data='deep1M'
data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}
result_path=./results
C=4096
K=100

for randomize in {0..2}
do
if [ $randomize == "1" ]
then 
    echo "IVF++"
elif [ $randomize == "2" ]
then 
    echo "IVF+"
    
else
    echo "IVF"
fi

res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf_${C}_${randomize}.index"

query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/O.fvecs"


./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K}

done

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

proj_dim=32

randomize=5
res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf1_pca_${proj_dim}.index"

query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_pca_matrix_${proj_dim}.fvecs"
linear="${index_path}/linear_ivf_pca_${proj_dim}.log"

./cmake-build-debug/src/search_ivf_pca -d 1 -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear}

randomize=6
res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf2_pca_${proj_dim}.index"

./cmake-build-debug/src/search_ivf_pca -d 2 -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear}