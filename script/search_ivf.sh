
# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0

cd ..
data='gist'
data_path=/home/DATA/vector_data/${data}
index_path=/home/DATA/centroid_data/${data}
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

res="${result_path}/${data}_IVF${C}_${randomize}.log"
index="${index_path}/${data}_ivf_${C}_${randomize}.index"

query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${data_path}/O.fvecs"


./cmake-build-debug/src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K}

done
