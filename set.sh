#export datasets=("sift" "gist" "deep1M" "_word2vec" "_glove2.2m" "_tiny5m")
export datasets=("_msong")
export store_path=/home/DATA/vector_data

# the operation to determine use SSE define in ./src/search_hnsw.cpp ./src/search_ivf.cpp