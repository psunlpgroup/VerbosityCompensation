import csv
import json
import os

import nltk
import string

add_option = True


def preprocess(text):
  text = nltk.word_tokenize(text)
  if len(text) == 0:
    return ""
  if text[-1] in string.punctuation:
    text = text[:-1]
  return ' '.join(text)

name_list = []
dataset = []

for name in os.listdir("PROJECT_PATH/dataset/MMLU/data/test"):
    file_path = f"PROJECT_PATH/dataset/MMLU/data/test/{name}"
    with open(file_path) as file:
        reader = csv.reader(file)
        for line in reader:
            options = line[1:-1]
            output = ord(line[-1]) - ord('A')
            answer = preprocess(options[output])
            length = len(nltk.word_tokenize(answer))
            if length > 3:
                continue
            if 'true' in answer.lower() or 'false' in answer.lower():
                continue
            turns = ["",""]
            if add_option:
                turns = ["Here are some candidates as hints:"] + options
            dataset.append({'raw': [name]+line, 'turns': turns, 'question':line[0], 'output': answer})

    dataset_name = name.replace('.csv','')
    if add_option:
        dataset_name += "+op"

# Prepare to save
save_file_path = f"dataset/mmlu.json"
new_data = dataset
import random
random.seed(42)
if len(new_data) > 500:
    new_data = random.sample(new_data, 500)
with open(save_file_path, 'w') as file:
    json.dump(new_data, file)
    print(f"Dump {len(new_data)} samples to {save_file_path}")
name_list.append(f"mmlu_{name.replace('.csv','')}")
print(' '.join(name_list))
print(name_list)