
cd ..

C=4096
pca_dim=32
opq_dim=96

for data in sift
do
echo "Indexing - ${data}"

if [ $data == "_tiny5m" ]
then
    opq_dim=96
elif [ $data == "_msong" ]
then
    opq_dim=105
elif [ $data == "_word2vec" ]
then
    opq_dim=75
elif [ $data == "_glove2.2m" ]
then
    opq_dim=75
elif [ $data == "gist" ]
then
    opq_dim=120
elif [ $data == "deep1M" ]
then
    opq_dim=64
elif [ $data == "sift" ]
then
    opq_dim=32
fi

python ./data/ivf.py -d ${data} -m "O"

python ./data/ivf.py -d ${data} -m "opq" -p ${opq_dim}

python ./data/ivf.py -d ${data} -m "pca" -p ${pca_dim}


for adaptive in {0..2}
do

echo "Indexing - ${data}"

data_path=/home/DATA/vector_data/${data}
index_path=./DATA/${data}

if [ $adaptive == "0" ] # raw vectors 
then
    data_file="${data_path}/${data}_base.fvecs"
    centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
else
    data_file="${index_path}/O${data}_base.fvecs"
    centroid_file="${index_path}/O${data}_centroid_${C}.fvecs"
fi

# 0 - IVF, 1 - IVF++, 2 - IVF+
index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"


./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

done

data_file="${index_path}/${data}_base_pca_${pca_dim}.fvecs"
centroid_file="${index_path}/${data}_centroid_pca_${pca_dim}.fvecs"
index_file="${index_path}/${data}_ivf2_pca_${pca_dim}.index"
adaptive=2
./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive


data_file="${index_path}/${data}_base_pca_${pca_dim}.fvecs"
centroid_file="${index_path}/${data}_centroid_pca_${pca_dim}.fvecs"
index_file="${index_path}/${data}_ivf1_pca_${pca_dim}.index"
adaptive=1
./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

data_file="${index_path}/${data}_base_opq_${opq_dim}.fvecs"
centroid_file="${index_path}/${data}_centroid_opq_${opq_dim}.fvecs"
index_file="${index_path}/${data}_ivf_opq_${opq_dim}.index"
adaptive=0
./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive

done