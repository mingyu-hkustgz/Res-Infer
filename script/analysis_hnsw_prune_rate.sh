cd ..
source set.sh

K=20
opq_dim=120
pca_dim=32
efSearch=500
opq_recall=0.995
pca_recall=0.995
data="gist"
data_path=${store_path}/${data}
pre_data=./DATA/${data}
index_type="hnsw1"
learn="${data_path}/${data}_learn.fvecs"
ground="${data_path}/${data}_learn_groundtruth.ivecs"

for opq_dim in {32,64,96,120,240};
do

  index="${pre_data}/${data}_ef500_M16_opq_${opq_dim}.index"
  trans="${pre_data}/${data}_opq_matrix_${opq_dim}.fvecs"
  code_book="${pre_data}/${data}_codebook_${opq_dim}.centroid"

  linear="${pre_data}/linear/linear_${index_type}_opq_${opq_dim}_${K}.log"
  res="./logger/prune_rate.log"

  ./cmake-build-debug/src/analysis_hnsw_opq -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -r ${res} -b ${code_book} -k ${K} -s ${efSearch} -e ${opq_recall} -d 1

done

trans="${pre_data}/${data}_pca_matrix_${pca_dim}.fvecs"
index="${pre_data}/${data}_ef500_M16_pca.index"
linear="${pre_data}/linear/linear_${index_type}_pca_${pca_dim}_${K}.log"
res="./logger/prune_rate.log"

./cmake-build-debug/src/analysis_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -r ${res} -k ${K} -s ${efSearch} -e ${pca_recall} -d 1


trans="${pre_data}/O.fvecs"
index="${pre_data}/O${data}_ef500_M16.index"
linear="${pre_data}/linear/linear_${index_type}_poj_${pca_dim}_${K}.log"
res="./logger/prune_rate.log"

./cmake-build-debug/src/analysis_hnsw_pca -i ${index} -q ${learn} -g ${ground} -t ${trans} -l ${linear} -r ${res} -k ${K} -s ${efSearch} -e ${pca_recall} -d 1


