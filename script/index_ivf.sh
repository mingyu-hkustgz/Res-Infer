cd ..
source set.sh
C=4096
pca_dim=32
opq_dim=96

for data in "${datasets[@]}"; do
  echo "Indexing - ${data}"


  python ./data/ivf.py -d ${data} -m "naive"

  python ./data/ivf.py -d ${data} -m "O"

  for adaptive in {0..1}
  do

  echo "Indexing - ${data}"

  data_path=${store_path}/${data}
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
done
