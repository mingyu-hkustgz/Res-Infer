#export datasets=("gist" "deep1M" "_glove2.2m" "_tiny5m" "_sift10m" "_word2vec")
export datasets=("sift100m")
export store_path=$HOME/DATA/vector_data
# the operation to determine use SSE define in ./src/search_hnsw.cpp ./src/search_ivf.cpp