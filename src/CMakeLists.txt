set(CMAKE_CXX_STANDARD 14)

add_executable(index_hnsw index_hnsw.cpp)

add_executable(index_ivf index_ivf.cpp)

add_executable(search_hnsw search_hnsw.cpp)

add_executable(search_ivf search_ivf.cpp)

add_executable(binery_search_parameter binery_search_parameter.cpp)

add_executable(logger_hnsw_pca logger_hnsw_pca.cpp)

add_executable(logger_ivf_pca logger_ivf_pca.cpp)

add_executable(logger_hnsw_opq logger_hnsw_opq.cpp)

add_executable(logger_ivf_opq logger_ivf_opq.cpp)

add_executable(test_seperate_log test_seperate_log.cpp)

add_executable(test_learned_inference test_learned_inference.cpp)

add_executable(search_hnsw_sse search_hnsw.cpp)
target_compile_definitions(search_hnsw_sse PRIVATE USE_SSE)


add_executable(search_hnsw_avx search_hnsw.cpp)
target_compile_definitions(search_hnsw_avx PRIVATE USE_SSE USE_AVX)

add_executable(search_hnsw_avx512 search_hnsw.cpp)
target_compile_definitions(search_hnsw_avx512 PRIVATE USE_SSE USE_AVX USE_AVX512)

add_executable(search_ivf_sse search_ivf.cpp)
target_compile_definitions(search_ivf_sse PRIVATE USE_SSE)

add_executable(search_ivf_avx search_ivf.cpp)
target_compile_definitions(search_ivf_avx PRIVATE USE_SSE USE_AVX)

add_executable(search_ivf_avx512 search_ivf.cpp)
target_compile_definitions(search_ivf_avx512 PRIVATE USE_SSE USE_AVX USE_AVX512)