import json

import nltk

if __name__ == '__main__':
  for position in [14]:
    source_path = f"PROJECT_PATH/dataset/lost-in-the-middle/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_{position}.jsonl"
    data = []
    with open(source_path) as file:
      for line in file:
        sample = json.loads(line)
        data.append(sample)
    new_data = []
    import random
    random.seed(42)
    data = random.sample(data, 500)

    for sample in data:
      new_sample = {"raw":sample}
      new_sample['turns'] = [x['title'] + '\n\n' + x['text'] + '\n' for x in sample['ctxs']]
      new_sample['output'] = sample['answers'][0]
      new_sample['question'] = sample['question']
      length = len(nltk.word_tokenize(new_sample['output']))
      if length > 3:
        continue
      new_data.append(new_sample)
    import os
    save_file_path = f"dataset/nq_open_{position}.json"
    with open(save_file_path, 'w') as file:
      json.dump(new_data, file)

    print(f"Dump {len(new_data)} samples to {save_file_path}")

