from datasets import load_dataset
import json
import evaluate
from nltk.tokenize import sent_tokenize
datasett = load_dataset("jordiclive/wikipedia-summary-dataset",split="train[80%:81%]")
prompts = datasett
sprinter = open('sprinter.jsonl',)
data = json.load(sprinter)
generated_responses_sprinter = []
for i in range(100):
   generated_responses_sprinter.append(data[i]['response'])
sprinter.close()
# loading reference files
ref_summary = []
prompt_index = 0
for prompt_single in prompts:
  short_prompt = prompt_single['summary']
  ref_summary.append(short_prompt)
  prompt_index += 1
  if prompt_index == 100:
    break
  
rouge_score = evaluate.load("rouge")
rouge_score.compute(
        predictions=generated_responses_sprinter,
        references=ref_summary,
        use_stemmer=True,

)
