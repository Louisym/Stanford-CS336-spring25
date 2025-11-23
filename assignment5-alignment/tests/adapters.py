from __future__ import annotations

import os
from typing import Any, Callable, Literal
import einops
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    assert len(prompt_strs) == len(output_strs), 'invalid input or label dimensions!'
    input_prompts_ids, output_ids = [], []
    for p in prompt_strs:
        p_id = tokenizer.encode(p, add_special_tokens=False)
        input_prompts_ids.append(torch.tensor(p_id))
    for o in output_strs:
        o_id = tokenizer.encode(o, add_special_tokens=False)
        output_ids.append(torch.tensor(o_id))
    prompt_and_output_lens = [len(promp)+len(out) for promp, out in zip(input_prompts_ids, output_ids)]
    D_output = max(prompt_and_output_lens) - 1
    #padding
    paded_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    input_ids = []
    labels = []
    response_mask = []
    for p_id, o_id in zip(input_prompts_ids, output_ids):
        input_id = torch.cat((p_id, o_id, torch.tensor([tokenizer.eos_token_id])), dim=-1)
        response_m = torch.cat((torch.zeros_like(p_id).to(dtype=torch.bool), torch.ones_like(o_id).to(dtype=torch.bool), torch.tensor([False])), dim=-1)
        slice_input_id = input_id[:-1]
        slice_output_id = input_id[1:]
        slice_response_m = response_m[1:]
        pad_len = D_output - slice_input_id.shape[0]

        padded_input_id = F.pad(slice_input_id, (0, pad_len), value=paded_val)
        padded_output_id = F.pad(slice_output_id, (0, pad_len), value=paded_val)
        response_mask_padded = F.pad(slice_response_m, (0, pad_len), value=False)
        
        input_ids.append(padded_input_id)
        labels.append(padded_output_id)
        response_mask.append(response_mask_padded)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'response_mask': torch.stack(response_mask)
    }






def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    assert len(rollout_responses) == len(repeated_ground_truths), 'invalid input with inequal # of labels and responses'

    # Compute raw rewards for each response
    raw_rewards_list = []
    for text_response, text_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(text_response, text_truth)
        r = reward_info['reward']
        raw_rewards_list.append(r)

    # Convert to tensor and reshape to (n_groups, group_size)
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    n_groups = len(rollout_responses) // group_size
    grouped_rewards = raw_rewards.reshape(n_groups, group_size)

    # Compute mean and std within each group (along dim=1)
    mean_r = torch.mean(grouped_rewards, dim=1, keepdim=True)  # shape: (n_groups, 1)
    std_r = torch.std(grouped_rewards, dim=1, keepdim=True)    # shape: (n_groups, 1)

    # Normalize within each group
    unnormalized_A = grouped_rewards - mean_r
    normalized_A = unnormalized_A / (std_r + advantage_eps)

    # Choose which advantages to return
    if normalize_by_std:
        A = normalized_A
    else:
        A = unnormalized_A

    # Flatten back to (rollout_batch_size,)
    A_flat = A.reshape(-1)
    raw_rewards_flat = raw_rewards

    return (
        A_flat,
        raw_rewards_flat,
        {
            'mean': mean_r.mean().item(),  # overall mean of group means
            'std': std_r.mean().item(),     # overall mean of group stds
            'min': raw_rewards.min().item(),
            'max': raw_rewards.max().item()
        }
    )



def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    normed_logits = F.softmax(logits, dim=-1)
    log_p = torch.log(normed_logits)
    return -torch.sum(normed_logits*log_p, dim=-1)

def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    pred_logits = model(input_ids).logits
    log_probs_all = F.log_softmax(pred_logits, dim=-1)

    # Gather the log probabilities of the actual tokens in labels
    # labels shape: (batch_size, seq_length)
    # log_probs_all shape: (batch_size, seq_length, vocab_size)
    # We need to gather along the vocab dimension
    labels_expanded = labels.unsqueeze(-1)  # (batch_size, seq_length, 1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expanded).squeeze(-1)

    if return_token_entropy:
        entropy = run_compute_entropy(pred_logits)
    else:
        entropy = None

    return {
        'log_probs': log_probs,
        'token_entropy': entropy
    }



def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    raw_rewards_or_advantages = einops.repeat(raw_rewards_or_advantages, 'b 1 -> b s', s=policy_log_probs.shape[-1])
    loss = - policy_log_probs*raw_rewards_or_advantages
    metadata = {}
    return (
        loss,
        metadata
    )


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    advantages = einops.repeat(advantages, 'b 1 -> b s', s=policy_log_probs.shape[-1])#broadcast manually
    cliprange = einops.repeat(torch.tensor(cliprange).unsqueeze(0), '1 -> b s', b=ratio.shape[0], s=ratio.shape[1])

    def clip_func(ratio, cliprange):
        clip_scaler = torch.where(ratio > 1.0+cliprange, 1.0+cliprange, ratio)
        clip_scaler = torch.where(clip_scaler < 1.0-cliprange, 1.0-cliprange, clip_scaler)
        return clip_scaler
    clip_scaler = clip_func(ratio, cliprange).to(device=advantages.device)
    return (- torch.min(ratio*advantages, clip_scaler*advantages),
            {
                'ratio': ratio,
                'clip_scaler': clip_scaler,
            }
            )



def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    assert loss_type in ("no_baseline", "reinforce_with_baseline", "grpo_clip"), 'wrong input loss type, please select one of "no_baseline", "reinforce_with_baseline", "grpo_clip"'
    if loss_type == 'no_baseline':
        assert raw_rewards is not None, 'please input raw_rewards'
        loss_info = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None, 'please input advantages'
        loss_info = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:
        assert advantages is not None and old_log_probs is not None and cliprange is not None, 'please fill in all required args, including advantages, old_log_probs and cliprange'
        loss_info = run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    loss = loss_info[0]
    metadata = loss_info[1]
    return (
        loss,
        metadata
    )



def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    #forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grad_tensor = torch.zeros_like(policy_log_probs, device=device)
    
    cross_entropy = run_masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant)
    loss = - cross_entropy.mean(dim=-1)
    loss /= gradient_accumulation_steps
    loss.backward()
    

    return (
        loss,
        {
            'loss': loss.detach() 
        }
    )



    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss_per_toekn, metadata = run_compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    l_mean = run_masked_mean(loss_per_toekn, response_mask, ) #batch dimension's loss
    l_mean /=  gradient_accumulation_steps
    l_mean.backward()
    return (
        l_mean,
        metadata
            )



def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    assert normalize_constant != 0, 'invalid constant forr normalization!'
    sum_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return sum_tensor.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
