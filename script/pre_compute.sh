cd ..
source set.sh

pca_dim=32
nbits=8

for data in "${datasets[@]}"; do
  echo "Indexing - ${data}"
  python ./data/randomized.py -d ${data}
done
