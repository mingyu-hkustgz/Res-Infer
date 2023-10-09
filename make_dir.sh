mkdir ./DATA
mkdir ./results
mkdir ./results/recall@20
mkdir ./results/recall@100


for dataset in {gist,deep1M};
do
  mkdir ./DATA/${dataset}
  mkdir ./DATA/${dataset}/linear
  mkdir ./results/recall@20/${dataset}
  mkdir ./results/recall@100/${dataset}
done

mkdir ./logger
mkdir ./figure