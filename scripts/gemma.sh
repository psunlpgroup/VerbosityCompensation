for segment_length in 3000
do
  for dataset in nq_open_14 # scrolls_qasper scrolls_narrative_qa longbench
  do
    dataset_file="./preprocessed/dataset/${dataset}.json"
    agent_model='google/gemma-7b-it'
    exp_name="${segment_length}"
    export CUDA_VISIBLE_DEVICES=0,7
    echo "$dataset_file $exp_name"
    python gemma.py \
     --dataset_file ${dataset_file} \
     --segment_length ${segment_length} \
     --exp_name ${exp_name} \
     --agent_model ${agent_model}
  done
done
