for segment_length in 96000
do
  for agent_model in gpt-4o-2024-05-13
  do
    for dataset in nq_open_14 # scrolls_qasper longbench scrolls_narrative_qa
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
