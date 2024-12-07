for segment_length in 150000
do
  for agent_model in claude-3-5-sonnet-20240620
  do
    for dataset in nq_open_14 # scrolls_qasper scrolls_narrative_qa longbench
    do
      dataset_file="./preprocessed/dataset/${dataset}.json"
      exp_name="${segment_length}"
      echo "$dataset_file $exp_name"
      python claude.py \
       --dataset_file ${dataset_file} \
       --segment_length ${segment_length} \
       --exp_name ${exp_name} \
       --agent_model ${agent_model}
    done
  done
done
