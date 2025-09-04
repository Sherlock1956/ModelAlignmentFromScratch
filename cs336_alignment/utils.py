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
    response_mask = []
    input_ids = []
    labels = []
    for i in range(len(prompt_strs)):
        input_id = tokenizer.encode(prompt_strs[i])
        output_id = tokenizer.encode(output_strs[i])
        input_id_full = input_id + output_id
        local_len = len(input_id) + len(output_id)
        prompt_and_output_lens.append(local_len)
        mask = [0.0] * (local_len - 1)
        mask[len(input_id) - 1:] = [1.0] * len(output_id)
        response_mask.append(mask)
        input_ids.append(input_id_full[:-1])
        labels.append(input_id_full[1:])
    max_len = max(prompt_and_output_lens)
    # 问题出现在，需要在加了padding之后再将full的token_ids进行截断第一个和最后一个，而不是先截断再加padding!
    for i in range(len(prompt_strs)):
        if prompt_and_output_lens[i] < max_len:
            padding_num = max_len - prompt_and_output_lens[i]
            input_ids[i] = input_ids[i] + [tokenizer.pad_token_id] * padding_num
            labels[i] = labels[i] + [tokenizer.pad_token_id] * padding_num
            response_mask[i] = response_mask[i] + [0.0] * padding_num
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    response_mask = torch.tensor(response_mask)
    return {
        "input_ids": input_ids.to(torch.long),
        "labels": labels.to(torch.long),
        "response_mask": response_mask.to(torch.bool)
    }
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/Users/lyx/Downloads/Study/projects/python/CS336-assignment5/models/Qwen2.5-Math-1.5B/qwen/Qwen2___5-Math-1___5B")
    prompt_strs = [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]
    output_strs = [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]
    tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
