cd ..

data='deep1M'
pca_dim=32
opq_dim=64
nbits=8

python ./data/randomized.py -d ${data}

python ./data/pca.py -d ${data} -p ${pca_dim}

python ./data/opq.py -d ${data} -m ${opq_dim} -b ${nbits}
