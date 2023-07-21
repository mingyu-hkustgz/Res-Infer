

cd ..

data='gist'
data_path=/home/DATA/vector_data/${data}
index_path=/home/DATA/graph_data/hnsw
result_path=./results/

ef=500
M=16

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

res="${result_path}/${data}_ef${ef}_M${M}_${randomize}.log"
query="${data_path}/${data}_query.fvecs"
gnd="${data_path}/${data}_groundtruth.ivecs"
trans="${data_path}/O.fvecs"

./cmake-build-debug/src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans}

done


