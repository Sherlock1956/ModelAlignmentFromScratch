from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from vllm import LLM, SamplingParams
from typing import List
import random
try:
    from drgrpo_grader import r1_zero_reward_fn
    import utils
    from math_baseline import evaluate
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
    from .math_baseline import evaluate
# 设置日志级别，减少 VLLM 的输出
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
def get_response(
    vllm_model: LLM,
    prompts: List[str],
    eval_sampling_params
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    res = [output.outputs for output in outputs]
    return res
# model prepare
device = 'cuda' if torch.cuda.is_available() else 'mps'
model_path = "models/Qwen2.5-0.5B/qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_path
)
model = model.to(device)
llm = LLM(model=model_path, gpu_memory_utilization=0.4)
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"],n=4
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,)
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-6)
# dataset prepare
reward_fn = r1_zero_reward_fn
r1_zero_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
gsm8k = []
with open("data/gsm8k/train.jsonl") as f:
    lines = f.readlines()
    for line in lines:
        gsm8k.append(json.loads(line))
prompts_to_be_filtered = []
answer_to_be_filtered = []
prompts_filtered = []
answer_filtered = []
for dict in gsm8k:
    prompts_to_be_filtered.append(r1_zero_prompt.format(question=dict['question']))
    answer_to_be_filtered.append(dict['answer'][dict['answer'].find("####") + 5:])
n_ei_step = 5
local_step = 0
for i in range(n_ei_step):
    # sample correct data from gsm8k
    print(f"ei step: {i + 1}")
    load_policy_into_vllm_instance(model, llm)
    indices = list(range(len(prompts_to_be_filtered)))
    indices = random.sample(indices, k=1024)
    prompts_to_be_filtered_part = [prompts_to_be_filtered[i] for i in indices]
    answer_to_be_filtered_part = [answer_to_be_filtered[i] for i in indices]
    outputs = get_response(llm, prompts_to_be_filtered_part, sampling_params)
    for j in range(len(outputs)):
        for k in range(len(outputs[j])):
            result = reward_fn(outputs[j][k].text, answer_to_be_filtered_part[j])
            if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
                prompts_filtered.append(prompts_to_be_filtered_part[j])
                answer_filtered.append(outputs[j][k].text)
    print(f"correct answer: {len(prompts_filtered)}")
    # train step
    epoch = 3
    batch_size = 8
    micro_batch_size = 1
    gradient_accumulation_steps = batch_size // micro_batch_size # 4
    log_directory = 'cs336_alignment/ei_logs'
    writer = SummaryWriter(log_directory)
    for j in range(epoch):
        pbar = tqdm(range(len(prompts_filtered) // micro_batch_size), desc=f"Epoch {j+1}/{epoch}")
        for j in pbar:
            prompt_strs = prompts_filtered[j * micro_batch_size:j * micro_batch_size + micro_batch_size]
            answer_strs = answer_filtered[j * micro_batch_size:j * micro_batch_size + micro_batch_size]
            train_batch = utils.tokenize_prompt_and_output(prompt_strs, answer_strs, tokenizer)
            result_dict = utils.get_response_log_probs(model, train_batch['input_ids'].to(device), train_batch['labels'].to(device))
            log_probs = result_dict['log_probs']
            loss, log_info = utils.sft_microbatch_train_step(log_probs, train_batch['response_mask'].to(device),gradient_accumulation_steps)
            writer.add_scalar('train/loss', loss.item(), local_step)
            if (local_step + 1) % gradient_accumulation_steps == 0 or len(prompts_filtered) < batch_size:
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新进度条显示损失
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Step': local_step})
            
            if (local_step <= 500 and local_step % 100 == 0) or (local_step <= 2000 and local_step % 500 == 0) or (local_step > 2000 and local_step % 1000 == 0):
                save_directory = f'{log_directory}/{local_step}'
                os.makedirs(save_directory, exist_ok=True)
                load_policy_into_vllm_instance(model, llm)
                accuracy, type1_num, type2_num, type3_num = evaluate(save_directory,llm)
                print(f"accuracy on test data at training step {local_step} is {accuracy}")
                writer.add_scalar('val/accuracy', accuracy, local_step)
                writer.add_scalar('val/type1', type1_num, local_step)
                writer.add_scalar('val/type2', type2_num, local_step)
                writer.add_scalar('val/type3', type3_num, local_step)
            local_step += 1

