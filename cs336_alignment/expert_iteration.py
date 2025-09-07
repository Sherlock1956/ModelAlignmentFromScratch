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
    from math_baseline import evaluate_vllm, evaluate
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
    from .math_baseline import evaluate_vllm
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
# model prepare
device = 'cuda' if torch.cuda.is_available() else 'mps'
model_path = "models/Qwen2.5-0.5B/qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_path
)
model = model.to(device)
llm = LLM(model=model_path, gpu_memory_utilization=0.3)
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
prompts = []
answer = []
for dict in gsm8k:
    prompts.append(r1_zero_prompt.format(question=dict['question']))
    answer.append(" " + dict['answer'].replace("#### "," </think> <answer> ") + " </answer>")
# train step
epoch = 3
batch_size = 8
micro_batch_size = 2
gradient_accumulation_steps = batch_size // micro_batch_size # 4
local_step = 0
log_directory = 'cs336_alignment/sft_logs'
writer = SummaryWriter(log_directory)
for i in range(epoch):
    pbar = tqdm(range(len(prompts) // micro_batch_size), desc=f"Epoch {i+1}/{epoch}")
    for j in pbar:
        prompt_strs = prompts[j * micro_batch_size:j * micro_batch_size + micro_batch_size]
        answer_strs = answer[j * micro_batch_size:j * micro_batch_size + micro_batch_size]
        train_batch = utils.tokenize_prompt_and_output(prompt_strs, answer_strs, tokenizer)
        result_dict = utils.get_response_log_probs(model, train_batch['input_ids'].to(device), train_batch['labels'].to(device))
        log_probs = result_dict['log_probs']
        loss, log_info = utils.sft_microbatch_train_step(log_probs, train_batch['response_mask'].to(device),gradient_accumulation_steps)
        writer.add_scalar('train/loss', loss.item(), local_step)
        if (local_step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 更新进度条显示损失
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Step': local_step})
        
        if local_step % 1000 == 0:
            save_directory = f'{log_directory}/{local_step}'
            os.makedirs(save_directory, exist_ok=True)
            # model.save_pretrained(save_directory=save_directory)
            # tokenizer.save_pretrained(save_directory=save_directory)
            load_policy_into_vllm_instance(model, llm)
            accuracy, type1_num, type2_num, type3_num = evaluate(save_directory,llm)
            print(f"accuracy on test data at training step {local_step} is {accuracy}")
            writer.add_scalar('val/accuracy', accuracy, local_step)
            writer.add_scalar('val/type1', type1_num, local_step)
            writer.add_scalar('val/type2', type2_num, local_step)
            writer.add_scalar('val/type3', type3_num, local_step)
            # 额外的显存清理，确保彻底释放
            gc.collect()
            torch.cuda.empty_cache()
        local_step += 1

