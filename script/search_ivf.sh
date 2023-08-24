
# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0

opq_dim=32
pca_dim=32
efSearch=50
C=4096
K=100
cd ..
for data in gist
do
echo "Searching - ${data}"

if [ $data == "_tiny5m" ]
then
    opq_dim=96
    efSearch=100
elif [ $data == "_msong" ]
then
    opq_dim=105
    efSearch=25
elif [ $data == "_word2vec" ]
then
    opq_dim=75
    efSearch=20
elif [ $data == "_glove2.2m" ]
then
    opq_dim=75
    efSearch=100
elif [ $data == "gist" ]
then
    opq_dim=120
    efSearch=50
elif [ $data == "deep1M" ]
then
    opq_dim=64
    efSearch=50
elif [ $data == "sift" ]
then
    opq_dim=32
    efSearch=30
fi

data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}
result_path="./results/recall@${K}/${data}"

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


./cmake-build-debug/src/search_ivf -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -s ${efSearch}

done


for randomize in {3..4}
do

res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf_opq_${opq_dim}.index"
linear="${index_path}/linear_ivf_opq_${opq_dim}_${K}.log"
query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_opq_matrix_${opq_dim}.fvecs"
code_book="${index_path}/${data}_codebook_${opq_dim}.centroid"

./cmake-build-debug/src/search_ivf -d ${randomize} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -b ${code_book} -l ${linear} -s ${efSearch}

done


randomize=5
res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf1_pca_${pca_dim}.index"

query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${index_path}/${data}_pca_matrix_${pca_dim}.fvecs"
linear="${index_path}/linear_ivf_pca_${pca_dim}_${K}.log"

./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear} -s ${efSearch}

randomize=6
res="${result_path}/${data}_ad_ivf_${randomize}.log"
index="${index_path}/${data}_ivf2_pca_${pca_dim}.index"

./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${linear} -s ${efSearch}

done