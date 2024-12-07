import argparse
import json
import os.path
import os
import time

import torch
import nltk
from openai import OpenAI
from retrying import retry
from tqdm import tqdm

from metrics.metric_lib.f1 import *
from metrics.metric_lib.rouge import *
import metrics.metric_lib.longbench as long_eval

from typing import List, Optional
from nltk import word_tokenize

API_KEY = "USER_KEY"
USE_COT = True

class GPT:
    def __init__(self, model_name: str="gpt-3.5-turbo-0125", ):
        self.client = OpenAI(api_key=API_KEY,)
        self.model = model_name
    def response(self, input_example, max_new_tokens=512, do_sample=False):
        #@retry(wait_exponential_multiplier=10000, wait_exponential_max=160001)
        # Wait 2^x * 10,000 milliseconds between each retry, up to 160 seconds, then 160 seconds afterwards
        def retry_create():
            # These parameters follow the guideline at https://github.com/openai/openai-python
            return self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_example,
                    }
                ],
                model=self.model,
                temperature=0,
                max_tokens=max_new_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=1
            )

        output = retry_create()
        # time.sleep(time_sleep)  # to slow down the requests
        tmp = output.choices[0].message.content.strip()

        return tmp, []



def _split_list_to_segment(long_input: List[str], segment_length: int,
                           count_function=lambda x: len(word_tokenize(x)),
                           duplication=1) -> List[List[str]]:
  """
  Split the long source dialogue into segments that are accepable by model.
  Each segment is a task for an agent.
  :param long_input: a list of sting, for document it is sentences, for
  dialogue, it is the turns in meeting transcript with the format of
  speaker: content
  :param duplication: copy and paste each segment for multiple times default is 1
  :return: A list of segment, each segment is a list of strings. The meaning
  of string is the same as input.

  """
  # First, compute how many tokens are there in each turn
  token_counter = []

  for turn in long_input:
    token_counter.append(count_function(turn))

  # Then, split the source according to its turn
  segments = []
  segment_counter = []
  current_token = 0
  current_segment = []
  for turn, token_count in zip(long_input, token_counter):
    if current_token >= segment_length:
      segment_counter.append(current_token)
      current_token = 0
      segments.append(current_segment)
      current_segment = []
    current_segment.append(turn)
    current_token += token_count

  if current_token != 0:
    segment_counter.append(current_token)
    current_token = 0
    segments.append(current_segment)
    current_segment = []

  segments = segments * duplication
  return segments

def run(args):
  data_file = args.dataset_file
  agent_model = args.agent_model
  exp_name = args.exp_name
  segment_length = args.segment_length
  reduction_rate = args.length_reduction_rate
  use_cot = USE_COT
  # Prepare result log_gpt3.5 file
  dataset_name = data_file.split('/')[-1].split('.')[0].strip()
  result_dir = f"./result/{dataset_name}/{agent_model.replace('/','_')}"
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  # Load dataset
  with open(os.path.join(data_file)) as file:
    data = json.load(file)
  tasks = data

  print("Loading model!")
  model = GPT(model_name=agent_model)

  scores = []
  retry_counter = 0
  with open(os.path.join(result_dir, exp_name+'.json'), 'w') as file:
    for idx, task in enumerate(tqdm(tasks)):
      if use_cot and idx > 100:
          break
      print("Processing Task", idx)
      prefix = "You are given an article and a question. Answer the question as concisely as you can, using a single phrase if possible. Article:\n"  # If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\nArticle:\n"
      # prefix = "Article:\n"
      question_clarification = "\nQuestion:\n"
      requirement = "\nUsing a single phrase rather than a sentence. Please answer in 3 words. Do not repeat any question related information or explain the answer.\nThe answer is:\n"
      if use_cot:
        prefix = "You are given an article and a question. Article:\n"  # If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\nArticle:\n"
        requirement = "\n\nLet's think step by step. Use as few steps as possible. And for each step, use less than 10 tokens. Answer as concisely as possible. Steps:"

      question = task['question']
      total_length = segment_length
      reduction_length = max(1, int(total_length * reduction_rate))
      is_retry = False

      while 1:
        try:
          task['segments'] = _split_list_to_segment(task['turns'], total_length)
          source = '\n'.join(task['segments'][0]) +'\n' # for truncation model, we only use the first segment
          input_prompt = prefix + source + question_clarification + question + requirement
          summary, output_distribution = model.response(input_prompt)
          print("Success, total length:", total_length)
          break
        except Exception as e:
            if "quota" in e.__repr__(): # Seems this won't triger
                print("Quota Exceeded, Retry:", e)
                time.sleep(40)
            else:
                total_length -= reduction_length
                if total_length < 0:
                    print("Length is 0")
                    summary, output_distribution = "", []
                    break
                print("Too long, retry. Total length:", total_length)
            is_retry = True
      retry_counter += is_retry
      file.write(json.dumps({"input":input_prompt,"summary":summary, "distribution":output_distribution})+'\n')
      print("pred:", summary)
      gold = task['output']
      print("gold:", gold)

      # Evaluation
      if dataset_name == "longbench_repobench-p":
        results = long_eval.code_sim_score(summary, gold)
        print("Edit Distance:", results)
      elif dataset_name == "longbench_hotpotqa":
        results = long_eval.qa_f1_score(summary, gold)
        print("F1", results)
      elif dataset_name == "longbench_musique":
        results = long_eval.qa_f1_score(summary, gold)
        print("F1", results)
      elif dataset_name == 'quality':
        results = gold in summary
      elif dataset_name == 'scrolls_quality':
        gold_letter = gold.split(')')[0].strip() + ')'
        gold_content = gold.split(')')[1].strip()
        results = gold_letter in summary or gold_content in summary
      elif dataset_name == 'scrolls_qmsum' or dataset_name == 'scrolls_gov_report' \
          or dataset_name == 'scrolls_summ_screen_fd':
        results = compute_rouge([summary], [gold])
        r1 = results['rouge1'][0].fmeasure
        r2 = results['rouge2'][0].fmeasure
        rl = results['rougeL'][0].fmeasure
        results = (r1*r2*rl)**(1/3)
      else:
        results = compute_f1([summary], [[gold]])
      print("Score:", results)
      scores.append(results)
    print("Average Score:", sum(scores)/len(scores))
    print("Longer than 1 segment:", retry_counter)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", default="./preprocessed/dataset/scrolls_qasper.json", type=str)
    parser.add_argument("--agent_model", default="gpt-3.5-turbo-0125", type=str)
    parser.add_argument("--segment_length", default=12000, type=int)
    parser.add_argument("--exp_name", default='12000_cot', type=str)
    parser.add_argument("--length_reduction_rate", default=0.05)
    args = parser.parse_args()

    run(args)