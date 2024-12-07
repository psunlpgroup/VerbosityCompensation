import string

import nltk
import os

model_name = {
  "gemma": "google_gemma-7b-it",
  "gemma-2-9b": "google_gemma-2-9b-it",
  "gemma-2-27b": "google_gemma-2-27b-it",
  "llama-3-8b": "meta-llama_Meta-Llama-3-8B-Instruct",
  "llama-3-70b": "meta-llama_Meta-Llama-3-70B-Instruct",
  'mistral-7b': "mistralai_Mistral-7B-Instruct-v0.3",
  'mixtral-8x7b': "mistralai_Mixtral-8x7B-Instruct-v0.1",
  'gpt-3.5': "gpt-3.5-turbo-0125",
  'gpt-4o': 'gpt-4o-2024-05-13',
  'gemini-flash': "gemini-1.5-flash",
  'claude-haiku': "claude-3-haiku-20240307",
  'claude-sonnet': "claude-3-5-sonnet-20240620",
  'gemini-1.0-pro':   'gemini-1.0-pro',
  'gemini-1.5-pro': 'gemini-1.5-pro'

}


exp_name = {
  "gemma": "3000_0607_3",
  "gemma-2-9b": "6000",
  "gemma-2-27b": "6000",
  "llama-3-8b": "6000",
  "llama-3-70b": "6000",
  'mistral-7b': "6000",
  'mixtral-8x7b': "6000",
  'gpt-3.5': "12000",
  'gpt-4o': "96000",
  'gemini-flash': '750000',
  'gemini-1.0-pro': '24000',
  'gemini-1.5-pro':'1500000',
  'claude-haiku': '150000',
  'claude-sonnet': '150000'
}

long_names = ['scrolls_qasper','scrolls_narrative_qa','longbench',]
mmlu_names = ['mmlu', 'mmlu_sociology_test', 'mmlu_machine_learning_test', 'mmlu_miscellaneous_test', 'mmlu_high_school_macroeconomics_test', 'mmlu_conceptual_physics_test', 'mmlu_moral_disputes_test', 'mmlu_world_religions_test', 'mmlu_college_chemistry_test', 'mmlu_astronomy_test', 'mmlu_elementary_mathematics_test', 'mmlu_high_school_statistics_test', 'mmlu_virology_test', 'mmlu_professional_accounting_test', 'mmlu_security_studies_test', 'mmlu_college_physics_test', 'mmlu_anatomy_test', 'mmlu_global_facts_test', 'mmlu_medical_genetics_test', 'mmlu_clinical_knowledge_test', 'mmlu_high_school_microeconomics_test', 'mmlu_abstract_algebra_test', 'mmlu_international_law_test', 'mmlu_high_school_geography_test', 'mmlu_professional_law_test', 'mmlu_college_medicine_test', 'mmlu_high_school_computer_science_test', 'mmlu_management_test', 'mmlu_business_ethics_test', 'mmlu_marketing_test', 'mmlu_nutrition_test', 'mmlu_college_computer_science_test', 'mmlu_high_school_psychology_test', 'mmlu_computer_security_test', 'mmlu_college_mathematics_test', 'mmlu_high_school_european_history_test', 'mmlu_high_school_biology_test', 'mmlu_high_school_us_history_test', 'mmlu_econometrics_test', 'mmlu_prehistory_test', 'mmlu_electrical_engineering_test', 'mmlu_moral_scenarios_test', 'mmlu_professional_medicine_test', 'mmlu_high_school_government_and_politics_test', 'mmlu_college_biology_test', 'mmlu_us_foreign_policy_test', 'mmlu_high_school_chemistry_test', 'mmlu_public_relations_test', 'mmlu_human_aging_test', 'mmlu_professional_psychology_test', 'mmlu_philosophy_test', 'mmlu_high_school_world_history_test', 'mmlu_high_school_mathematics_test', 'mmlu_human_sexuality_test', 'mmlu_jurisprudence_test', 'mmlu_high_school_physics_test', 'mmlu_logical_fallacies_test', 'mmlu_formal_logic_test']
mmlu_option_names = [x+'+op' for x in mmlu_names]
lost_names = ['nq_open_14']
reveral_curse_name = ['rc_ask_parent']
traverse_dataset_names = long_names + mmlu_names + mmlu_option_names + lost_names + reveral_curse_name
traverse_models = model_name.keys()
eval_single = False # Will print more samples if eval_single is true

########### Options ############
eval_type = 'single' # ['all', 'set', 'single']
use_cot = False

if eval_type == 'set':
  ## Overwrite Eval Set ##
  traverse_dataset_names = ['mmlu']
  traverse_models = model_name.keys()

if eval_type == 'single':
  ## Eval Single ##
  eval_single = True
  dataset_name_eval = ['longbench']
  model_short_name_eval = "mixtral-8x7b"
################################

def preprocess(text):
  text = text.split('\n')[0]
  text = text.replace('**','')
  text = nltk.word_tokenize(text)
  if len(text) == 0:
    return ""
  # if text[-1] in string.punctuation:
  #   text = text[:-1]
  return ' '.join(text)

################ TRAVERSE ###################
for dataset_name in traverse_dataset_names:
  for model_short_name in traverse_models:
    if eval_single:
      if dataset_name not in dataset_name_eval or model_short_name_eval != model_short_name:
        continue
    print(dataset_name, model_name[model_short_name])
    if use_cot:
      exp_name[model_short_name] += '_cot'
    vanilla_prediction_file = f"../result/{dataset_name}/{model_name[model_short_name]}/{exp_name[model_short_name]}.json"
    if not os.path.exists(vanilla_prediction_file):
      continue
    source = f"../preprocessed/dataset/{dataset_name}.json"
    import json
    import metrics.metric_lib.longbench as long_eval
    with open(source) as file:
      data = json.load(file)

    def cal_distribution(dist):
      def var(data):
        mean = sum(data) / len(data)
        population_variance = sum((x - mean) ** 2 for x in data) / len(data)
        return population_variance
      # return var([x[1] for x in dist])
      return dist[0][1]

    def read_gemini(file_path):
      pred_data = []
      with open(file_path, 'r') as file:
        source = []
        for line in file:
          source.append(line)
      if len(source) == 1: # if there is only one line
        source = source[0]
        source = source[1:-1].replace("\"}{\"", '"\"}  {\""')
        source = source.split("}  {")
        for line in source:
          line = line.split('\"summary\":')
          line = line[-1][2:-2].replace('<ctrl100>','').split('\\')[0]
          pred_data.append(line)
      else:
        for s in source:
          pred_data.append(json.loads(s))
      return pred_data
    vanilla_data = read_gemini(vanilla_prediction_file)
    gold_thres = list(range(100))#[2,3] #

    for thres in range(4,5):
      shorter_scores = []
      longer_scores = []
      idx = -1
      count_short = 0
      for vanilla_sample, sample_source in zip(vanilla_data, data):
        vanilla = vanilla_sample['summary'] if isinstance(vanilla_sample, dict) else vanilla_sample
        gold = sample_source['output']
        idx += 1
        if 'true' in gold.lower() or 'false' in gold.lower():
          continue
        vanilla = preprocess(vanilla) # "**" removed due to markdown format of Gemini
        gold = preprocess(gold)
        vanilla_results = long_eval.qa_f1_score(vanilla, gold)
        vanilla_p, vanilla_r, vanilla_f = long_eval.qa_prf_score(vanilla, gold)

        vanilla_length = len(nltk.word_tokenize(vanilla))
        gold_length = len(nltk.word_tokenize(gold))

        # # This is for analysing vaiance
        # variance = cal_distribution(vanilla_sample['distribution'][0])
        # vanilla_r = variance

        if gold_length not in gold_thres:
          continue
        if thres > vanilla_length:
          # print(sample_source['question'])
          # print("gold", gold, "\nvanilla", vanilla, vanilla_r, vanilla_f)

          shorter_scores.append(vanilla_r)
          count_short += 1
        else:

          longer_scores.append(vanilla_r)
          if eval_single:
            print("sample id", idx)
            print(sample_source['question'])
            print("gold", gold, "\nvanilla", vanilla.split('\n')[0], vanilla_r, vanilla_f)
            print()

      diff = sum(shorter_scores)/max(len(shorter_scores),1) - sum(longer_scores)/max(len(longer_scores),1)
      print("shorter - longer", diff)
      print("shorter",sum(shorter_scores)/max(len(shorter_scores),1), count_short)
      avg = lambda x: sum(x)/max(len(x),1)
      print("longer", sum(longer_scores)/max(len(longer_scores),1),len(longer_scores))
      print("Frequency", len(longer_scores)/len(longer_scores+shorter_scores))
      print("Avg Perfromance:", avg(shorter_scores+longer_scores))
      print("Total samples:", count_short + len(longer_scores),'\n')


