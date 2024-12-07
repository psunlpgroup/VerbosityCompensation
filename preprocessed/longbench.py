import json
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
if __name__ == '__main__':
  dataset_names = ['2wikimqa', 'hotpotqa','multifieldqa_en','musique'] #,'qasper','triviaqa']
  save_file_path = f"dataset/longbench.json"
  new_data = []
  for dataset_name in dataset_names:
    raw_data_file_path = f"PROJECT_PATH/dataset/longbench/{dataset_name}.jsonl"
    with open(raw_data_file_path) as file:
      for line in file:
        sample = json.loads(line)
        new_sample = {}
        new_sample['raw'] = sample
        new_sample['turns'] = sample['context'].split('\n') # Most of the lines are empty but it is okay!
        if len(sample['answers']) > 1:
          # print("Error: more than 1 answers detected") # Error will be found in multifield-qa
          continue
        new_sample['output'] = sample['answers'][0] # it is a list, directly use 1
        new_sample['question'] = sample['input'] # Query
        new_sample['dataset_name'] = 'longbench_' + dataset_name

        if new_sample['output'][-1] == '.': # Remove last period
          new_sample['output'] = new_sample['output'][:-1]

        word_list = nltk.word_tokenize(new_sample['output'])

        if len(word_list) not in [1,2,3]:
          continue
        if word_list[0].lower() in stopwords.words('english'):
          continue
        if len(word_list) == 1 and word_list[0].lower() in ["yes", "no", "unanswerable"]:
          continue
        print(new_sample['output'])
        new_data.append(new_sample)

  with open(save_file_path, 'w') as file:
    json.dump(new_data, file)

  print(f"Dump {len(new_data)} samples to {save_file_path}")


