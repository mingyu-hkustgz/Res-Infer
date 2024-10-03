cd ..
source set.sh

efConstruction=500
M=16

for data in "${datasets[@]}"; do

  echo "precompute - ${data}"
  log_file="./logger/${data}/OPQ-Train-time.log"
  start_time=$(date +%s)
  python ./data/opq.py -d ${data}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "OPQ Train time: ${duration}(s)" | tee -a ${log_file}

  echo "Indexing - HNSW - ${data}"

  data_path=${store_path}/${data}
  index_path=./DATA/${data}
  pre_path=./DATA/${data}

  data_file="${pre_path}/${data}_base_opq.fvecs"
  index_file="${index_path}/${data}_ef${efConstruction}_M${M}_opq.index"
  log_file="./logger/${data}/OPQ-HNSW-time.log"
  start_time=$(date +%s)
  ./cmake-build-debug/src/index_hnsw -d $data_file -i $index_file -e $efConstruction -m $M
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW Index time: ${duration}(s)" | tee -a ${log_file}


  echo "Indexing -IVF- ${data}"

  python ./data/ivf.py -d ${data} -m "opq"

  data_file="${index_path}/${data}_base_opq.fvecs"
  centroid_file="${index_path}/${data}_centroid_opq.fvecs"
  index_file="${index_path}/${data}_ivf_opq.index"
  adaptive=0

  log_file="./logger/${data}/OPQ-HNSW-time.log"
  start_time=$(date +%s)
  ./cmake-build-debug/src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "IVF Index time: ${duration}(s)" | tee -a ${log_file}

done
