for segment_length in 12000
do
  for agent_model in gpt-3.5-turbo-0125
  do
    for dataset in scrolls_qasper # nq_open_14  scrolls_narrative_qa longbench
    do
      dataset_file="./preprocessed/dataset/${dataset}.json"
      exp_name="${segment_length}"
      echo "$dataset_file $exp_name"
      python gpt.py \
       --dataset_file ${dataset_file} \
       --segment_length ${segment_length} \
       --exp_name ${exp_name} \
       --agent_model ${agent_model}
    done
  done
done
