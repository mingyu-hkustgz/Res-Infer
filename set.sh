# export datasets=("gist" "deep1M" "_glove2.2m" "_tiny5m" "_sift10m" "_word2vec" "_msong")
#export datasets=("deep1M" "_msong" "_glove2.2m" "_tiny5m" "_word2vec" "gist")
#export datasets=("sift20m" "sift40m" "sift60m" "sift80m" "sift100m")
export datasets=("gist")
export store_path=$HOME/DATA/vector_data
# the operation to determine use SSE define in ./src/search_hnsw.cpp ./src/search_ivf.cpp