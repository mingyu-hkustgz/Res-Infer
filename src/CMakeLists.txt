set(CMAKE_CXX_STANDARD 14)

add_executable(index_hnsw index_hnsw.cpp)

add_executable(index_ivf index_ivf.cpp)

add_executable(search_hnsw search_hnsw.cpp)

add_executable(search_ivf search_ivf.cpp)

add_executable(binery_search_parameter binery_search_parameter.cpp)

add_executable(logger_hnsw_pca logger_hnsw_pca.cpp)

add_executable(logger_ivf_pca logger_ivf_pca.cpp)

add_executable(test_seperate_log test_seperate_log.cpp)

add_executable(test_hnsw_log test_hnsw_log.cpp)
