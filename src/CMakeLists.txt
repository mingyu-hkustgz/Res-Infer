set(CMAKE_CXX_STANDARD 14)

add_executable(index_hnsw index_hnsw.cpp)

add_executable(index_ivf index_ivf.cpp)

add_executable(search_hnsw search_hnsw.cpp)

add_executable(search_ivf search_ivf.cpp)

add_executable(logger_hnsw logger_hnsw.cpp)

add_executable(logger_ivf logger_ivf.cpp)

add_executable(test_pq test_pq.cpp)

add_executable(test_pca test_pca.cpp)
