from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
import gc
from tqdm import tqdm
from tensorboardX import SummaryWriter
from vllm import LLM, SamplingParams
try:
    from drgrpo_grader import r1_zero_reward_fn
    import utils
    from math_baseline import evaluate
    from .config import *
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
    from .math_baseline import  evaluate
    from config import *
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

# prepare model
device = 'cuda' if torch.cuda.is_available() else 'mps'
model_path = "models/Qwen2.5-0.5B/qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_path
)
model = model.to(device)
llm = LLM(model=model_path, gpu_memory_utilization=0.3)
tokenizer = AutoTokenizer.from_pretrained(model_path)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,)

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
prompts_to_be_filtered = []
answer_to_be_filtered = []
prompts_filtered = []
answer_filtered = []
for dict in gsm8k:
    prompts_to_be_filtered.append(r1_zero_prompt.format(question=dict['question']))
    answer_to_be_filtered.append(dict['answer'][dict['answer'].find("####") + 5:])

for i in range(n_grpo_steps):
    # repeat grpo sampling and traning for n_grpo_steps times
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    # sample n_prompts_per_rollout_batch from the training dataset
    pass
    # generate group_size answer for each prompt, get rollout_batch_size answer in total
    pass
    # generate old_policy_log_probabilities if using off_policy
    pass
    for j in range(epochs_per_rollout_batch):
        micro_train_batch_size = train_batch_size // gradient_accumulation_steps
        n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    # compute group normalized advantages and raw_reward
    utils.compute_group_normalized_rewards()
    # compute policy_gradient_loss with respect to loss_type
    loss, info = utils.compute_policy_gradient_loss()