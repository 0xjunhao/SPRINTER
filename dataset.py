
from huggingface_hub import login
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import transformers
import numpy as np
import torch
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os


def load_models(device, target_name, draft_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(target_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    target = transformers.AutoModelForCausalLM.from_pretrained(target_name)
    target.to(device)
    target.eval()

    tokenizer_draft = transformers.AutoTokenizer.from_pretrained(draft_name)

    if tokenizer_draft.pad_token_id is None:
        tokenizer_draft.pad_token_id = tokenizer_draft.eos_token_id

    draft = transformers.AutoModelForCausalLM.from_pretrained(draft_name)
    draft.to(device)
    draft.eval()
    return tokenizer, target, tokenizer_draft, draft


def load_hugg_dataset(dataset_name, split_name):
    datasett = load_dataset(dataset_name, split=split_name)
    return datasett


def collect_ver_data(datasett, tokenizer, target, tokenizer_draft, draft, file_name, begin, end, word_num):
    collection = []
    i = 0
    sentence_length = []

    for i in tqdm(range(begin, end)):
        # prompt = datasett[i]['full_text'].split()
        prompt = datasett[i]['text'].split()
        num_tok = np.ceil(0.3*len(prompt))
        prompt = prompt[:int(num_tok)]
        pro = ''
        for y in prompt:
            pro = pro + y + ' '
        prompt = pro
        word_count = 0
        while word_count < word_num:

            model_inputs = tokenizer(
                prompt, truncation=True, max_length=1024,  return_tensors='pt').to(device)
            input_length = model_inputs.input_ids.shape[1]
            model_inputs_draft = tokenizer_draft(
                prompt,  truncation=True, max_length=1024, return_tensors='pt').to(device)

            if input_length > 1024:
                print(
                    f"Warning: Input length {input_length} is greater than 1024, but should be truncated by tokenizer.")
                continue
            # generate token from draft

            greedy_output_draft = draft.generate(**model_inputs_draft, max_new_tokens=1, output_scores=True,
                                                 return_dict_in_generate=True, output_hidden_states=True, max_length=1024)
            transition_scores_draft = draft.compute_transition_scores(
                greedy_output_draft.sequences, greedy_output_draft.scores, normalize_logits=True)
            generated_tokens_draft = greedy_output_draft.sequences[:,
                                                                   input_length:]
            probs_draft = torch.nn.functional.softmax(
                greedy_output_draft.scores[0][0])

            # generate token from target

            greedy_output_target = target.generate(
                **model_inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True, output_hidden_states=True, max_length=1024)
            transition_scores_target = target.compute_transition_scores(
                greedy_output_target.sequences, greedy_output_target.scores, normalize_logits=True)
            generated_tokens_target = greedy_output_target.sequences[:, input_length:]
            probs_target = torch.nn.functional.softmax(
                greedy_output_target.scores[0][0])

            q_x = probs_draft[generated_tokens_draft.item()]
            p_x = probs_target[generated_tokens_draft.item()]

            if q_x <= p_x*1 and word_count == 0:
                report = 1
                new_token = tokenizer_draft.decode(
                    generated_tokens_draft.item())
                collection.append({'prompt': prompt, 'token': generated_tokens_draft.item(
                ), 'threshold_label': report, 'Type': "S0"})
            elif q_x <= p_x*1 and word_count != 0:
                report = 1
                new_token = tokenizer_draft.decode(
                    generated_tokens_draft.item())
                collection.append({'prompt': prompt, 'token': generated_tokens_draft.item(
                ), 'threshold_label': report,  'Type': "S"})
            else:
                report = 0
                new_token = tokenizer.decode(generated_tokens_target.item())
                collection.append({'prompt': prompt, 'token': generated_tokens_draft.item(
                ), 'threshold_label': report, 'Type': "L"})

            prompt = prompt + new_token
            word_count += 1
        pd.DataFrame(collection).to_csv(file_name)
        i += 1


if __name__ == "__main__":
    load_dotenv()
    token_name = os.getenv("HF_TOKEN")  # expects HF_TOKEN in your .env file
    dataset_name = "billion-word-benchmark/lm1b"
    split_name = "train"
    target_name = 'EleutherAI/gpt-neo-1.3B'
    draft_name = 'EleutherAI/gpt-neo-125m'
    file_name = 'collection_length03_ratio1_lm1b_neo.csv'
    begin = 0
    end = 500
    word_num = 20
    login(token=token_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, target, tokenizer_draft, draft = load_models(
        device, target_name, draft_name)
    datasett = load_hugg_dataset(dataset_name, split_name)
    collect_ver_data(datasett, tokenizer, target, tokenizer_draft,
                     draft, file_name, begin, end, word_num)
