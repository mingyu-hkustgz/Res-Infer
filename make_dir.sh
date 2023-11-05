source set.sh
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

mkdir ./logger
mkdir ./figure

for dataset in "${datasets[@]}";
do
  echo $dataset
  mkdir ./DATA/${dataset}
  mkdir ./DATA/${dataset}/linear
  mkdir ./results/recall@20/${dataset}
  mkdir ./results/recall@100/${dataset}
  mkdir ./figure/${dataset}
done