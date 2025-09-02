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
    prompt_len = len(prompt_strs)
    output_len = len(output_strs)
    input_ids = tokenizer.tokenize(prompt_strs)
    output_ids = tokenizer.tokenize(output_strs)
    input_ids_full = torch.concat((input_ids, output_ids),dim=-1)
    input_ids = input_ids_full[:-1]
    labels = input_ids_full[1:]
    response_mask = torch.zeros_like(labels)
    response_mask[:prompt_len - 1] = 1.0
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
