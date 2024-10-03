cd ..
source set.sh

pca_dim=32
nbits=8

for data in "${datasets[@]}"; do
  echo "Indexing - ${data}"
  log_file="./logger/${data}/ADS-random-time.log"
  start_time=$(date +%s)
  python ./data/randomized.py -d ${data}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "ADS random time: ${duration}(s)" | tee -a ${log_file}
done
