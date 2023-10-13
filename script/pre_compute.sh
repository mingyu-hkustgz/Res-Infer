cd ..

pca_dim=32
opq_dim=75
nbits=8

for data in {sift,gist,deep1M,_word2vec,_glove2.2m}; do
echo "Indexing - ${data}"

if [ $data == "_tiny5m" ]
then
    opq_dim=48
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
    opq_dim=32
elif [ $data == "sift" ]
then
    opq_dim=16
fi

python ./data/randomized.py -d ${data}

python ./data/pca.py -d ${data} -p ${pca_dim}

python ./data/opq.py -d ${data} -m ${opq_dim} -b ${nbits}

done
