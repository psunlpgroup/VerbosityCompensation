for segment_length in 1500000
do
  for agent_model in  gemini-1.5-pro
  do
    for dataset in nq_open_14 # scrolls_qasper longbench scrolls_narrative_qa
    do
      dataset_file="./preprocessed/dataset/${dataset}.json"
      exp_name="${segment_length}"
      api_key=API_KEY  # Rui
      echo "$dataset_file $exp_name"
      python gemini.py \
       --api_key ${api_key} \
       --dataset_file ${dataset_file} \
       --segment_length ${segment_length} \
       --exp_name ${exp_name} \
       --agent_model ${agent_model}
    done
  done
done
