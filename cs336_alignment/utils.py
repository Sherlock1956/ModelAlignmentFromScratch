from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
    prompt_strs: list[str] List of prompt strings.
    output_strs: list[str] List of output strings.
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys:
    input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
    labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token. 
    response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask on the response tokens in the labels."""
    prompt_and_output_lens = []
    prompt_len = []
    response_mask = []
    input_ids = []
    labels = []
    for i in range(len(prompt_strs)):
        input_id = tokenizer.encode(prompt_strs[i])
        output_id = tokenizer.encode(output_strs[i])
        input_id_full = input_id + output_id
        local_len = len(input_id) + len(output_id)
        prompt_len.append(len(input_id))
        prompt_and_output_lens.append(local_len)
        mask = [0.0] * (local_len - 1)
        # mask[len(input_id) - 1:] = [1.0] * len(output_id)
        response_mask.append(mask)
        input_ids.append(input_id_full)
        labels.append(input_id_full)
    max_len = max(prompt_and_output_lens)
    # 问题出现在，需要在加了padding之后再将full的token_ids进行截断第一个和最后一个，而不是先截断再加padding!
    for i in range(len(prompt_strs)):
        if prompt_and_output_lens[i] < max_len:
            padding_num = max_len - prompt_and_output_lens[i]
            input_ids[i] = input_ids[i] + [tokenizer.pad_token_id] * padding_num
            labels[i] = labels[i] + [tokenizer.pad_token_id] * padding_num
            response_mask[i] = response_mask[i] + [0.0] * padding_num
            input_ids[i] = input_ids[i][:-1]
            labels[i] = labels[i][1:]
            response_mask[i][prompt_len[i]-1:prompt_and_output_lens[i]-1] = [1.0] * (prompt_and_output_lens[i]-prompt_len[i])
        else:
            input_ids[i] = input_ids[i][:-1]
            labels[i] = labels[i][1:]
            response_mask[i][prompt_len[i]-1:] = [1.0] * (prompt_and_output_lens[i]-prompt_len[i])
            pass
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    response_mask = torch.tensor(response_mask)
    return {
        "input_ids": input_ids.to(torch.long),
        "labels": labels.to(torch.long),
        "response_mask": response_mask.to(torch.bool)
    }
def compute_entropy(logits):
    """
    logits: (batch_size, seq_len, vocab_size)
    """
    # 这个log_softmax内部使用了logsumexp的技术，也就是减去最大值再计算softmax的技巧，防止数值上溢
    log_prob = torch.nn.functional.log_softmax(logits,dim=-1)
    prob = torch.exp(log_prob)
    return -(torch.sum(prob * log_prob,dim=-1))
def get_response_log_probs(
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)# batch,seq,vocab
    log_probs = log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        return {
            "log_probs": log_probs,# batch_size, seq_len
            "token_entropy": compute_entropy(logits)
        }
    else:
        return {
            "log_probs": log_probs
        }
def masked_normalize(
        tensor,
        mask,
        normalize_constant,
        dim
    ):
    """
    tensor: (batch_size, seq_len)
    mask: (batch_size, seq_len)
    """
    # 创建一个新的张量而不是原地修改
    masked_tensor = tensor * mask.float()
    if dim is not None:
        res = torch.sum(masked_tensor,dim=dim) / normalize_constant
    else:
        res = torch.sum(masked_tensor) / normalize_constant
    return res # (batch_size,)
def sft_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        normalize_constant = 1.0
    ):
    # 计算每个序列的响应长度
    response_lengths = response_mask.sum(dim=-1)  # (batch_size,)
    
    # 对每个序列按其响应长度归一化
    masked_log_probs = policy_log_probs * response_mask.float()  # (batch_size, seq_len)
    sequence_losses = masked_log_probs.sum(dim=-1) / response_lengths.clamp(min=1)  # (batch_size,)
    
    # 计算平均损失
    loss = -sequence_losses.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}
def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std,
    ):
    normalized_rewards = []
    unnormalized_rewards = []
    prompt_size = len(rollout_responses) // group_size
    for i in range(prompt_size):
        local_rewards = []
        for j in range(group_size):
            reward = reward_fn(rollout_responses[i*group_size+j],repeated_ground_truths[i*group_size+j])
            full_reward = reward['reward']
            local_rewards.append(full_reward)
        unnormalized_rewards.extend(local_rewards)
        local_rewards = torch.tensor(local_rewards)
        mean = sum(local_rewards) / len(local_rewards)
        local_rewards = local_rewards - mean
        if normalize_by_std:
            std = local_rewards.std()
            local_rewards = local_rewards / (std + advantage_eps)
        normalized_rewards.extend(local_rewards.tolist())
    normalized_rewards = torch.tensor(normalized_rewards)
    unnormalized_rewards = torch.tensor(unnormalized_rewards)
    return (normalized_rewards, unnormalized_rewards, {})        
def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages,
        policy_log_probs
):
    return -(raw_rewards_or_advantages) * policy_log_probs

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-Math-1.5B/qwen/Qwen2.5-Math-1.5B")
    input_ids = torch.randint(0,100,(1,10))
    labels = torch.randint(0,100,(1,10))
    get_response_log_probs(model,input_ids,labels,True)