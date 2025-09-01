try:
    from drgrpo_grader import r1_zero_reward_fn
except:
    from .drgrpo_grader import r1_zero_reward_fn
import json
from typing import List
from vllm import LLM, SamplingParams

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: callable,
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    return outputs
if __name__ == "__main__":
    SamplingParams = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    llm = LLM(model="models/")
    reward_fn = r1_zero_reward_fn
    r1_zero_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    gsm8k = []
    with open("data/gsm8k/test.jsonl") as f:
        lines = f.readlines()
        for line in lines:
            gsm8k.append(json.loads(line))
    prompts = []
    answer = []
    for dict in gsm8k:
        prompts.append(r1_zero_prompt.format(dict['question']))
        answer.append(dict['answer'])
    print(len(prompts))
    outputs = evaluate_vllm(llm, reward_fn, prompts, SamplingParams)
    acc = 0
    for i in range(len(outputs)):
        acc += reward_fn(outputs[i], answer[i])
