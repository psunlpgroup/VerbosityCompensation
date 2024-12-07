import json
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

if __name__ == '__main__':
  save_file_path = f"dataset/scrolls_narrative_qa.json"

  splits = ['validation']
  new_data = []
  for split in splits:
    raw_data_file_path = f"PROJECT_PATH/dataset/scrolls/narrative_qa/{split}.jsonl"
    with open(raw_data_file_path) as file:
      for idx, line in enumerate(file):
        sample = json.loads(line)
        new_sample = {}
        # new_sample['raw'] = sample # Do not save this field to save memory
        new_sample['turns'] = sample['input'].split('\n')[1:] # Most of the lines are empty but it is okay!
        new_sample['output'] = sample['output']


        if new_sample['output'][-1] == '.': # Remove last period
          new_sample['output'] = new_sample['output'][:-1]
        word_list = nltk.word_tokenize(new_sample['output'])
        if len(word_list) not in [1,2]:
          continue
        if word_list[0].lower() in stopwords.words('english'):
          continue
        if len(word_list) == 1 and word_list[0].lower() in ["yes", "no", "unanswerable"]:
          continue

        print(new_sample['output'])

        new_sample['question'] = sample['input'].split('\n')[0] # Query and corresponding choices
        new_sample['id'] = f"{split}_{idx}"
        new_data.append(new_sample)


  import random
  random.seed(42)
  if len(new_data) > 500:
    new_data = random.sample(new_data, 500)
  with open(save_file_path, 'w') as file:
    json.dump(new_data, file)

    print(f"Dump {len(new_data)} samples to {save_file_path}")


