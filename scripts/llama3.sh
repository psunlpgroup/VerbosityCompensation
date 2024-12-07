for segment_length in 6000
do
  for agent_model in meta-llama/Meta-Llama-3-70B-Instruct # meta-llama/Meta-Llama-3-8B-Instruct
  do
    for dataset in nq_open_14 mmlu # scrolls_qasper scrolls_narrative_qa longbench
    do
      dataset_file="./preprocessed/dataset/${dataset}.json"
      exp_name="${segment_length}"
      export CUDA_VISIBLE_DEVICES=2,3
      echo "$dataset_file $exp_name"
      python llama3.py \
       --dataset_file ${dataset_file} \
       --segment_length ${segment_length} \
       --exp_name ${exp_name} \
       --agent_model ${agent_model}
    done
  done
done
