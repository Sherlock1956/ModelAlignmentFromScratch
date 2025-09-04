from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
try:
    from drgrpo_grader import r1_zero_reward_fn
    import utils
except:
    from .drgrpo_grader import r1_zero_reward_fn
    from . import utils
# model prepare
device = 'cuda' if torch.cuda.is_available() else 'mps'
model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen2.5-Math-1.5B/qwen/Qwen2.5-Math-1.5B",
    )
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-1.5B/qwen/Qwen2.5-Math-1.5B")
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,)
# dataset prepare
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
    prompts.append(r1_zero_prompt.format(question=dict['question']))
    answer.append(" " + dict['answer'].replace("#### "," </think> <answer> ") + " </answer>")
# train step
epoch = 3
batch_size = 8
micro_batch_size = 2
gradient_accumulation_steps = batch_size // micro_batch_size # 4
for i in range(epoch):
    local_step = 0
    for j in range(len(prompts) // micro_batch_size + 1):
        prompt_strs = prompts[j * micro_batch_size:j * micro_batch_size + micro_batch_size]
        answer_strs = answer[j * micro_batch_size:j * micro_batch_size + micro_batch_size]
        train_batch = utils.tokenize_prompt_and_output(prompt_strs, answer_strs, tokenizer)
        result_dict = utils.get_response_log_probs(model, train_batch['input_ids'].to(device), train_batch['labels'].to(device))
        log_probs = result_dict['log_probs']
        loss, log_info = utils.sft_microbatch_train_step(log_probs, train_batch['response_mask'].to(device),gradient_accumulation_steps)
        if (local_step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        print(f"iteration: {j}, all_iteration: {len(prompts) // micro_batch_size + 1}, loss: {loss.item()}")
        local_step += 1
