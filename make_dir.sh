mkdir ./DATA
mkdir ./results
mkdir ./results/recall@20
mkdir ./results/recall@100
mkdir cmake-build-debug
cd cmake-build-debug
cmake ..
make clean
make -j 40
cd ..
for dataset in {sift,gist,deep1M,_word2vec,_glove2.2m};
do
  mkdir ./DATA/${dataset}
  mkdir ./DATA/${dataset}/linear
  mkdir ./results/recall@20/${dataset}
  mkdir ./results/recall@100/${dataset}
done

mkdir ./logger
mkdir ./figure