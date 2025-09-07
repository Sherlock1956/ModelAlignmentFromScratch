try:
    from drgrpo_grader import r1_zero_reward_fn
except:
    from .drgrpo_grader import r1_zero_reward_fn
import json
import logging
import os
import gc
import torch
from typing import List
from vllm import LLM, SamplingParams

# 设置日志级别，减少 VLLM 的输出
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: callable,
    prompts: List[str],
    eval_sampling_params
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    res = [output.outputs[0].text for output in outputs]
    return res
def evaluate(model_path):
    llm = None
    try:
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
        )
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        llm = LLM(model=model_path, gpu_memory_utilization=0.4)
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
            answer.append(dict['answer'][dict['answer'].find("####") + 5:])
        print(len(prompts))
        outputs = evaluate_vllm(llm, reward_fn, prompts, sampling_params)
        acc = 0
        type1_num = 0
        type2_num = 0
        type3_num = 0
        for i in range(len(outputs)):
            gsm8k[i]['outputs'] = "<think>" + outputs[i]
            result = reward_fn(outputs[i], answer[i])
            gsm8k[i]['result'] = result
            if result['format_reward'] == 1.0 and result['answer_reward'] == 1.0:
                type = 1
                type1_num += 1
            elif result['format_reward'] == 1.0 and result['answer_reward'] == 0.0:
                type = 2
                type2_num += 1
            else:
                type = 3
                type3_num += 1
            gsm8k[i]['type'] = type
            acc += result['reward']
        accuracy = acc / len(outputs)
        for i in range(len(gsm8k)):
            gsm8k[i]['outputs'] = "<think>" + outputs[i]
        with open(f"{model_path}/test_log.json",'w') as f:
            json.dump(gsm8k,f,indent=4)
        return accuracy, type1_num, type2_num, type3_num
    finally:
        # 显式释放 VLLM 实例和显存
        if llm is not None:
            try:
                # 尝试销毁 VLLM 实例
                del llm
            except:
                pass
        
        # 强制垃圾回收
        gc.collect()
        
        # 清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
if __name__ == "__main__":
    model_path = "models/Qwen2.5-Math-1.5B/qwen/Qwen2.5-Math-1.5B"
    evaluate(model_path)