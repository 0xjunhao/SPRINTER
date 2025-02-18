import torch
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import torch
import torch.nn as nn
import transformers
import numpy as np
import pandas as pd
import time
import json
from utils import*
from dataset import*

class TokenAcceptanceModel(torch.nn.Module):
    def __init__(self, input_size, tokenizer, gpt_neo_model_name='EleutherAI/gpt-neo-125m'): 
        super(TokenAcceptanceModel, self).__init__()
        self.tokenizer = tokenizer
        self.gpt_neo =  transformers.AutoModelForCausalLM.from_pretrained(gpt_neo_model_name)
        for param in self.gpt_neo.parameters():
            param.requires_grad = False  # Freeze GPT-Neo
        self.classifier = torch.nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.gpt_neo(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # Use the last token's hidden state
        return self.classifier(hidden_states)



def speculative_sampling_sprinter(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, DRTmodel : torch.nn.Module,
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:

    seq_len = prefix.shape[1]
    T = seq_len + max_len
    time_vec = []
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device

    device = target_model.device

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    total_time = 0
    candidate_avg_length = []
    # accc_plot_list = []
    while prefix.shape[1] < T:
      candidate_input_ids = prefix
      cum_acc_prob = 1
      assist_steps = 0
      input_ids = prefix
      stop_threshold = 0.9
      generation_target_SD = time.time()

      ## start the drafting process
      while True:
          # print("start drafting "+ str(assist_steps))
          x, total_draft_model_time, hidden_states_last_layer = approx_model_cache.generate(candidate_input_ids, 1)
          candidate_input_ids = x
          assist_steps += 1
          acc_prediction = DRTmodel.classifier(hidden_states_last_layer)
          ### Sprinter way
          if acc_prediction < 0.5:
             break

          if assist_steps > 19:
            break

      ## start the targeting process
      candidate_length =  candidate_input_ids.shape[1] - input_ids.shape[1]

      prefix_len = input_ids.shape[1]
      # model_input_ids = candidate_input_ids[:, -candidate_length - 1 :]
      _ = target_model_cache.generate(candidate_input_ids, 1)
      # for i in range(candidate_length): # only verify last token
      if random_seed:
          torch.manual_seed(random_seed)
      r = torch.rand(1, device = device)
      j = x[:, prefix_len + candidate_length-1]

      if r > (target_model_cache._prob_history[:, prefix_len + candidate_length - 2, j]) / (approx_model_cache._prob_history[:, prefix_len + candidate_length - 2, j]):
          # reject
          n = prefix_len +  candidate_length - 2
      else:
          n = prefix_len + candidate_length - 1
          accepted_count += 1
      candidate_avg_length.append(n - prefix_len+1)
      # print("Target select "+ str(accepted_count))
      assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

      generation_target_SD_fin = time.time() - generation_target_SD
      prefix = x[:, :n + 1]

      approx_model_cache.rollback(n+1)

      assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"

      if n < prefix_len + candidate_length - 1:
          # reject someone, sample from the pos n
          start =time.time()
          t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
          resample_SD = time.time() - start
          if verbose:
              print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
          resample_count += 1
          target_model_cache.rollback(n+1)
      else:
          # all approx model decoding accepted
          assert n == target_model_cache._prob_history.shape[1] - 1
          start= time.time()
          t = sample(target_model_cache._prob_history[:, -1, :])
          resample_SD = time.time() - start
          if verbose:
              print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
          target_sample_count += 1
          target_model_cache.rollback(n+2)


      prefix = torch.cat((prefix, t), dim=1)
      total_time += generation_target_SD_fin + resample_SD + total_draft_model_time
    if verbose:
      print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix, total_time, sum(candidate_avg_length)/len(candidate_avg_length)





if __name__ == "__main__":
    dataset = 'lm1b'
    save_model_name =  'DRT_length03_ratio15_lm1b_neo.pth'
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "billion-word-benchmark/lm1b"
    split_name = "train"
    target_name = 'EleutherAI/gpt-neo-1.3B'
    draft_name = 'EleutherAI/gpt-neo-125m'
    max_len = 20
    sprinter_collection = []
    json_file_name = 'sprinter.jsonl'
    datasett = load_hugg_dataset(dataset_name, split_name)
    _, target, tokenizer, draft = load_models(device, target_name, draft_name)
    DRTmodel_sprinter = TokenAcceptanceModel(input_size=768, tokenizer=tokenizer).to(device)
    DRTmodel_sprinter.classifier.load_state_dict(torch.load(save_model_name))
    name_list = ['sprinter'] 
    for name in name_list:
        gamma = 5
        top_k = 1
        top_p = 0.9
        random_seed = 1234
        verbose = False
        running_sum_DRT_accept = 0
        running_sum_DRT_reject = 0
        ground_truth_accept = 0
        ground_truth_reject = 0
        prompt_counter = 0
        stop_threhold = 100
        time_vec_SD = 0
        prompts = datasett
        single_sampling_time =[]
        avg_token_list = []

        for prompt_single in prompts:
            if dataset == 'wiki':
                short_prompt = prompt_single['full_text'].split()
            elif dataset == 'lm1b':
                short_prompt = prompt_single['text'].split()
            num_tok = len(short_prompt)
            num_tok = np.ceil(0.3*num_tok)
            short_prompt = short_prompt[:int(num_tok)]
            prompt = ''
            for y in short_prompt:
                prompt = prompt + y + ' '
            # print('prompt_counter', prompt_counter)
            #getting rid of last space
            prompt = prompt.rstrip()
            input_ids = tokenizer.encode(prompt, truncation=True, max_length=1024, return_tensors='pt').to(device)
            input_length = input_ids.shape[1]
            # print('ORIG PROMPT')
            # print(prompt)
            begin_sample_time = time.time()
            if name == 'sprinter':
                output, time_SD,avg_token = speculative_sampling_sprinter(input_ids, draft, target, DRTmodel_sprinter, max_len, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
            end_sample_time = time.time()
            avg_token_list.append(avg_token)
            single_sampling_time.append(end_sample_time - begin_sample_time)
            time_vec_SD+=time_SD
            dic = {'overall_time': [sum(single_sampling_time)], 'time_vec_SD': [time_vec_SD], 'avg_single_run_token':[sum(avg_token_list)/len(avg_token_list)],'avg_time':[sum(single_sampling_time)/len(single_sampling_time)], 'std_time':[np.std(single_sampling_time)]}
            pd.DataFrame(dic).to_csv('lm1b_neo_time_'+str(name)+'.csv', index=False)
            if name == 'sprinter':
                sprinter_collection.append({'prompt': prompt, 'res': tokenizer.decode(output[0],skip_special_tokens=True)[len(prompt):],'response':prompt +  tokenizer.decode(output[0],skip_special_tokens=True)[len(prompt):]})
            prompt_counter += 1
            if prompt_counter == stop_threhold:
                break

    
    with open(json_file_name, 'w') as outfile:
        json.dump(sprinter_collection, outfile, ensure_ascii=False)
