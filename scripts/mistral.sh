for segment_length in 6000
do
  for agent_model in mistralai/Mixtral-8x7B-Instruct-v0.1 # mistralai/Mistral-7B-Instruct-v0.3  mistralai/Mixtral-8x7B-Instruct-v0.1 mistralai/Mixtral-8x22B-Instruct-v0.1
  do
    for dataset in nq_open_14 # scrolls_qasper scrolls_narrative_qa longbench nq_open_14 mmlu
    do
      dataset_file="./preprocessed/dataset/${dataset}.json"
      exp_name="${segment_length}"
      echo "$agent_model $dataset_file $exp_name"
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
      python mistral.py \
       --dataset_file ${dataset_file} \
       --segment_length ${segment_length} \
       --exp_name ${exp_name} \
       --agent_model ${agent_model}
    done
  done
done
