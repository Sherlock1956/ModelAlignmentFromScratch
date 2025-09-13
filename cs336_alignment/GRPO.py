from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
from typing import List
from tqdm import tqdm
from tensorboardX import SummaryWriter
from vllm import LLM, SamplingParams
import random
try:
    from drgrpo_grader import r1_zero_reward_fn
    import utils
    from math_baseline import evaluate
    from config import *
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
    from .math_baseline import  evaluate
    from .config import *
# set logging level
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
# prepare model
device = 'cuda' if torch.cuda.is_available() else 'mps'
model_path = "models/Qwen2.5-0.5B/qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_path
)
model = model.to(device)
llm = LLM(model=model_path, gpu_memory_utilization=0.3)
sampling_params = SamplingParams(
    temperature=sampling_temperature, min_tokens=sampling_min_tokens, max_tokens=sampling_max_tokens, stop=["\n"],n=group_size
)
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True
tokenizer = AutoTokenizer.from_pretrained(model_path)
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,betas=betas)

# prepare dataset
reward_fn = r1_zero_reward_fn
r1_zero_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
gsm8k = []
with open("data/gsm8k/train.jsonl") as f:
    lines = f.readlines()
    for line in lines:
        gsm8k.append(json.loads(line))
prompts = []
answer = []
for dict in gsm8k:
    prompts.append(r1_zero_prompt.format(question=dict['question']))
    answer.append(dict['answer'][dict['answer'].find("####") + 5:])

global_step = 0
writer = SummaryWriter(f"cs336_alignment/grpo_{loss_type}")
# repeat grpo sampling and traning for n_grpo_steps times
for i in range(n_grpo_steps):
    print(f"grpo_steps: {i + 1}")
    load_policy_into_vllm_instance(model, llm)
    if (i + 1) % 5 == 0:
        # log validation rewards
        model_path = f"cs336_alignment/grpo_{loss_type}" + f"{global_step}"
        validation_reward, format_reward = evaluate(model_path=model_path, llm=llm)
        writer.add_scalar("val/reward",validation_reward,global_step)
        writer.add_scalar("val/fmt_reward",format_reward,global_step)
    n_prompts_per_rollout_batch = rollout_batch_size // group_size # 256 // 8 = 32
    # sample n_prompts_per_rollout_batch from the training dataset
    indices = list(range(len(prompts)))
    indices = random.sample(indices, k=n_prompts_per_rollout_batch)
    prompts_rollout_batch = [prompts[i] for i in indices]
    answers_rollout_batch = [answer[i] for i in indices]
    repeated_prompts_rollout_batch = []
    repeated_answers_rollout_batch = []
    for j in range(n_prompts_per_rollout_batch):
        for k in range(group_size):
            repeated_answers_rollout_batch.append(answers_rollout_batch[j])
            repeated_prompts_rollout_batch.append(prompts_rollout_batch[j])
    # generate group_size answer for each prompt, get rollout_batch_size answer in total
    response_rollout_batch_obj = get_response(llm, prompts_rollout_batch, sampling_params)
    response_rollout_batch = []
    for j in range(n_prompts_per_rollout_batch):
        for k in range(group_size):
            response_rollout_batch.append(response_rollout_batch_obj[j][k].text)
    # compute group normalized advantages and raw_reward
    advantages, raw_rewards, info = utils.compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=response_rollout_batch,
        repeated_ground_truths=repeated_answers_rollout_batch,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=use_std_normalization
    )
    print(f"average reward: {raw_rewards.mean()}")
    print(f"average format reward: {info['format_rewards'].mean()}")
    writer.add_scalar("train/reward",raw_rewards.mean(),global_step)
    writer.add_scalar("train/fmt_reward",info['format_rewards'].mean().mean(),global_step)
    # generate old_policy_log_probabilities if using off_policy
    # 一定要注意这里应该使用的是response_rollout_batch作为train_batch
    train_batch = utils.tokenize_prompt_and_output(repeated_prompts_rollout_batch, response_rollout_batch, tokenizer)
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps # 256 // 128 = 2
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size # 256 // 2 = 128
    old_policy_log_probs = []
    with torch.no_grad():
        for j in tqdm(range(n_microbatches_per_rollout_batch),desc="compute old policy probs"):
            input_ids = train_batch['input_ids'][j*micro_train_batch_size:j*micro_train_batch_size + micro_train_batch_size].to(device)
            labels = train_batch['labels'][j*micro_train_batch_size:j*micro_train_batch_size + micro_train_batch_size].to(device)
            old_policy_log_probs.append(utils.get_response_log_probs(model, input_ids, labels)['log_probs'])
    for j in range(epochs_per_rollout_batch):
        print(f"epoch: {j + 1}")
        local_step = 0
        for k in tqdm(range(n_microbatches_per_rollout_batch),desc="training"):
            # prepare data in a micro batch
            input_ids = train_batch['input_ids'][k * micro_train_batch_size : k*micro_train_batch_size + micro_train_batch_size].to(device)
            labels = train_batch['labels'][k * micro_train_batch_size : k*micro_train_batch_size + micro_train_batch_size].to(device)
            raw_reward = raw_rewards[k * micro_train_batch_size : k*micro_train_batch_size + micro_train_batch_size].to(device).unsqueeze(1)
            advantage = advantages[k * micro_train_batch_size : k*micro_train_batch_size + micro_train_batch_size].to(device).unsqueeze(1)
            response_mask = train_batch['response_mask'][k * micro_train_batch_size : k*micro_train_batch_size + micro_train_batch_size].to(device)
            # generate policy_log_probabilities 
            result_dict = utils.get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            # compute and backward loss with given policy_log_probs, loss_type, advantages, etc.
            loss, info = utils.grpo_microbatch_train_step(
                policy_log_probs=result_dict['log_probs'],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=raw_reward,
                advantages=advantage,
                old_log_probs=old_policy_log_probs[k],
                cliprange=0.1
            )
            # update local_step, optimizer.step() if (local_step + 1) % gradient_accumulation_steps == 0
            if (local_step + 1) % gradient_accumulation_steps == 0:
                gradient_norm = utils.My_gradient_clipping(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            # log some useful info such as loss, reward, entropy, etc.
            writer.add_scalar("train/token_etp",result_dict['token_entropy'], global_step)
            writer.add_scalar("train/gradient_norm",gradient_norm, global_step)

            local_step += 1
            global_step += 1