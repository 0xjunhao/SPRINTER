### from https://github.com/Soumya1612-Rasha/Transfer-Q


from tqdm import tqdm
import json
import argparse
import os
import numpy as np
import random
import openai
from openai import OpenAI
import time
import pdb

openai.api_key=''


SYSTEM_PROMPT = """[System]
You are an assistant for checking the quality of the answer. We would like to request your feedback on the performance of two AI assistants in response to completing the initial prefix. Please rate the accuracy and level of detail of their completions for each prefix. Your evaluation should consider factors such as the accuracy and level of detail of the completion. Bear in mind that the response might be truncated at the end due to length constraints. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""





USER_PROMPT = """[Question]
{question}

[The Start of Assistant 1's Answer]
{answer1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2}

[The End of Assistant 2's Answer]"""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name_red", default="run_outs/rebuttal/ultrafeedback/ours_5_0", type=str)
    parser.add_argument("--run_name_blue", default="run_outs/rebuttal/ultrafeedback/sft_0", type=str)

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    return args


def clean(text, sep="###"):
    result = text.split(sep)[0]
    return result if len(result) > 0 else " "


def gpt4_eval(sys_prompt: str, user_prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        # pdb.set_trace()
        return response.choices[0].message.content
    except Exception as ex:
        print(ex)
        time.sleep(3)
    return "error"


if __name__ == "__main__":
    args = get_args()

    path = os.path.join(f"{args.run_name_red}.jsonl")
    generations_red = json.load(open(path, "r"))

    path = os.path.join(f"{args.run_name_blue}.jsonl")
    generations_blue = json.load(open(path, "r"))

    # selected_indices = random.sample(range(len(generations_red)), 100)
    num_samples = min(100, min(len(generations_blue), len(generations_red)))
    selected_indices = range(num_samples)
    print(f"Selected {num_samples} samples")
    generations_red = [generations_red[i] for i in selected_indices]
    generations_blue = [generations_blue[i] for i in selected_indices]

    evaluations = []
    win = tie = lose = 0
    for red, blue in tqdm(zip(generations_red, generations_blue), total=len(generations_red)):
        prompt = red["prompt"]
        response_red = red["res"]   
        response_blue = blue["res"] 

        side = random.randint(0, 1)
        if side == 0:
            user_prompt = USER_PROMPT.format(question=prompt, answer1=response_red, answer2=response_blue)
        else:
            user_prompt = USER_PROMPT.format(question=prompt, answer1=response_blue, answer2=response_red)

        while True:
            content = gpt4_eval(sys_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            if content != "error":
                break

        try:
            score1, score2 = map(float, content.split("\n")[0].split(" "))
        except Exception:
            print(content)
            assert True
            score1, score2 = 0, 0

        if side == 1:
            score1, score2 = score2, score1

        evaluations.append(
            {
                "prompt": prompt,
                "red_answer": response_red,
                "blue_answer": response_blue,
                "red_score": score1,
                "blue_score": score2,
                "result": content,
            },
        )

        win += score1 > score2
        tie += score1 == score2
        lose += score1 < score2

        print(win, tie, lose)
        print(score1, score2)

    result = {
        "run_name_red": args.run_name_red,
        "run_name_blue": args.run_name_blue,
        "win": win,
        "tie": tie,
        "lose": lose,
        "evaluations": evaluations,
    }
    # print(result)

    # eval_path = os.path.join("gpt-4-evaluations", f"{args.run_name_red}_{args.run_name_blue}.json")
    # json.dump(result, open(eval_path, "w"), indent=2)
