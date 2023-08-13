set(CMAKE_CXX_STANDARD 14)

add_executable(index_hnsw index_hnsw.cpp)

add_executable(index_ivf index_ivf.cpp)

add_executable(search_hnsw search_hnsw.cpp)

add_executable(search_hnsw_opq search_hnsw_opq.cpp)

add_executable(search_hnsw_pca search_hnsw_pca.cpp)

add_executable(search_ivf search_ivf.cpp)

add_executable(search_ivf_opq search_ivf_opq.cpp)

add_executable(search_ivf_pca search_ivf_pca.cpp)

add_executable(logger_hnsw_pca logger_hnsw_pca.cpp)

add_executable(logger_ivf_pca logger_ivf_pca.cpp)

add_executable(logger_hnsw_opq logger_hnsw_opq.cpp)

add_executable(logger_ivf_opq logger_ivf_opq.cpp)

add_executable(test_pq test_pq.cpp)

add_executable(test_pca test_pca.cpp)

