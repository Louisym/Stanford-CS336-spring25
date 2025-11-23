
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.adapters import *

import typer
import gc
import logging
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Literal

import dotenv
import fire
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

@dataclass
class Trainconfig:
    #data config
    experiment_name_base = 'experiments'
    experiment_name = 'grpo'
    model_name: str = 'Qwen2.5-Math-1.5B'
    data_path: str = './data/math/train.jsonl'
    prompt_path: str = './cs336_alignment/prompts/r1_zero.prompt'
    #train config
    gradient_accumulation_steps: int = 128
    n_grpo_steps: int = 10
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
     # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = 'cuda:0'

    #eval config
    eval_steps: int = 5
    eval_device: str = 'cuda:1'

    #vllm sampling params
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop_tokens: list[str] = field(default_factory=lambda: ["</solution>"])
    include_stop_str_in_output: bool = True
    min_tokens: int = 4
    vllm_seed: int = 42

@dataclass
class Evalconfig:
    data_path: str = "./data/math/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: list[str] = field(default_factory=lambda: ["</solution>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True

class GRPODataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = self.train_answers[idx].strip()

        return prompt, cot, answer



app = typer.Typer()

# Helper functions
def load_math_data(data_path: str):
    """Load MATH dataset from jsonl file

    Expects 'question'/'answer' format.
    """
    import json
    questions = []
    answers = []

    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            questions.append(item['question'])
            answers.append(item['answer'])

    return questions, answers

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file"""
    with open(prompt_path, 'r') as f:
        return f.read()

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Initialize vLLM model for inference"""
    from unittest.mock import patch
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy, llm):
    """Load policy weights into vLLM instance"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def format_prompts(questions: list[str], prompt_template: str) -> list[str]:
    """Format questions with prompt template"""
    return [prompt_template.replace('{question}', q) for q in questions]

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn,
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> dict:
    """Evaluate model on validation set"""
    # Generate outputs
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Extract generated text
    generated_texts = [output.outputs[0].text for output in outputs]

    # Compute rewards
    total_reward = 0
    format_reward = 0
    answer_reward = 0

    for gen_text, gt in zip(generated_texts, ground_truths):
        reward_info = reward_fn(gen_text, gt)
        total_reward += reward_info['reward']
        format_reward += reward_info.get('format_reward', 0)
        answer_reward += reward_info.get('answer_reward', 0)

    n = len(prompts)
    return {
        'accuracy': total_reward / n if n > 0 else 0,
        'format_reward': format_reward / n if n > 0 else 0,
        'answer_reward': answer_reward / n if n > 0 else 0,
        'generated_texts': generated_texts,
    }

@app.command()
def train(
    # Data config
    model_name: str = 'Qwen2.5-Math-1.5B',
    data_path: str = './data/math/train.jsonl',
    val_data_path: str = './data/math/validation.jsonl',
    prompt_path: str = './cs336_alignment/prompts/r1_zero.prompt',
    model_path: str = '/data/a5-alignment/models/Qwen2.5-Math-1.5B',

    # Train config
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 128,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,

    # Eval config
    eval_steps: int = 5,
    eval_batch_size: int = 512,

    # Device config
    train_device: str = 'cuda:0',
    eval_device: str = 'cuda:1',

    # Wandb config
    wandb_project: str = 'cs336-grpo',
    wandb_run_name: str = 'grpo-run',

    # Other
    seed: int = 42,
    save_dir: str = './checkpoints/grpo',
):
    """Main GRPO training loop"""
    import os
    import json
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    # Set random seeds
    torch.manual_seed(seed)

    # Initialize wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=locals())

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Load data
    print("Loading training data...")
    train_questions, train_answers = load_math_data(data_path)
    print(f"Loaded {len(train_questions)} training examples")

    print("Loading validation data...")
    val_questions, val_answers = load_math_data(val_data_path)
    print(f"Loaded {len(val_questions)} validation examples")

    # Load prompt template
    prompt_template = load_prompt_template(prompt_path)

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy = policy.to(train_device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # Initialize vLLM for rollouts
    print("Initializing vLLM...")
    vllm_model = init_vllm(model_path, eval_device, seed, gpu_memory_utilization)

    # Sampling parameters
    rollout_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Sanity checks
    assert train_batch_size % gradient_accumulation_steps == 0
    assert rollout_batch_size % group_size == 0
    assert train_batch_size >= group_size

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Main training loop
    train_step = 0
    eval_step = 0

    for grpo_step in range(n_grpo_steps):
        print(f"\n{'='*60}")
        print(f"GRPO Step {grpo_step + 1}/{n_grpo_steps}")
        print(f"{'='*60}")

        # === Rollout Phase ===
        print("\n[Rollout Phase]")

        # Sample questions for this rollout batch
        import random
        rollout_indices = random.sample(range(len(train_questions)), n_prompts_per_rollout_batch)
        rollout_questions = [train_questions[i] for i in rollout_indices]
        rollout_ground_truths = [train_answers[i] for i in rollout_indices]

        # Format prompts
        rollout_prompts = format_prompts(rollout_questions, prompt_template)

        # Load current policy weights into vLLM
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_model)

        # Generate rollouts (G outputs per prompt)
        print(f"Generating {group_size} rollouts per prompt for {n_prompts_per_rollout_batch} prompts...")
        rollout_outputs = vllm_model.generate(rollout_prompts, rollout_sampling_params)

        # Extract all generated responses (flatten group outputs)
        all_responses = []
        repeated_ground_truths = []
        repeated_prompts = []

        for output, gt, prompt in zip(rollout_outputs, rollout_ground_truths, rollout_prompts):
            for generation in output.outputs:
                all_responses.append(generation.text)
                repeated_ground_truths.append(gt)
                repeated_prompts.append(prompt)

        print(f"Generated {len(all_responses)} total responses")

        # === Compute Rewards and Advantages ===
        print("\n[Computing Rewards]")
        advantages, raw_rewards, reward_metadata = run_compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        print(f"Reward stats - Mean: {reward_metadata['mean']:.4f}, "
              f"Std: {reward_metadata['std']:.4f}, "
              f"Min: {reward_metadata['min']:.4f}, "
              f"Max: {reward_metadata['max']:.4f}")

        # Convert to tensors and reshape
        advantages = advantages.unsqueeze(1)  # (rollout_batch_size, 1)
        raw_rewards = raw_rewards.unsqueeze(1)  # (rollout_batch_size, 1)

        # === Tokenize rollouts ===
        print("\n[Tokenizing rollouts]")
        tokenized_data = run_tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=all_responses,
            tokenizer=tokenizer,
        )

        input_ids = tokenized_data['input_ids']  # (rollout_batch_size, seq_len)
        labels = tokenized_data['labels']  # (rollout_batch_size, seq_len)
        response_mask = tokenized_data['response_mask']  # (rollout_batch_size, seq_len)

        # Get old log probs for off-policy (if using GRPO-Clip)
        old_log_probs = None
        if loss_type == "grpo_clip":
            print("Computing old policy log probs...")
            with torch.inference_mode():
                old_log_prob_dict = run_get_response_log_probs(
                    model=policy,
                    input_ids=input_ids.to(train_device),
                    labels=labels.to(train_device),
                    return_token_entropy=False,
                )
                old_log_probs = old_log_prob_dict['log_probs'].detach()

        # === Training Phase ===
        print(f"\n[Training Phase - {epochs_per_rollout_batch} epoch(s)]")
        policy.train()

        for epoch in range(epochs_per_rollout_batch):
            # Shuffle data for this epoch
            perm = torch.randperm(rollout_batch_size)
            input_ids_epoch = input_ids[perm]
            labels_epoch = labels[perm]
            response_mask_epoch = response_mask[perm]
            advantages_epoch = advantages[perm]
            raw_rewards_epoch = raw_rewards[perm]
            if old_log_probs is not None:
                old_log_probs_epoch = old_log_probs[perm]
            else:
                old_log_probs_epoch = None

            # Iterate over minibatches
            n_batches = rollout_batch_size // micro_train_batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size

                # Get microbatch
                batch_input_ids = input_ids_epoch[start_idx:end_idx].to(train_device)
                batch_labels = labels_epoch[start_idx:end_idx].to(train_device)
                batch_response_mask = response_mask_epoch[start_idx:end_idx].to(train_device)
                batch_advantages = advantages_epoch[start_idx:end_idx].to(train_device)
                batch_raw_rewards = raw_rewards_epoch[start_idx:end_idx].to(train_device)
                batch_old_log_probs = old_log_probs_epoch[start_idx:end_idx].to(train_device) if old_log_probs_epoch is not None else None

                # Get policy log probs
                policy_log_prob_dict = run_get_response_log_probs(
                    model=policy,
                    input_ids=batch_input_ids,
                    labels=batch_labels,
                    return_token_entropy=True,
                )
                policy_log_probs = policy_log_prob_dict['log_probs']
                token_entropy = policy_log_prob_dict['token_entropy']

                # Compute loss and backprop
                loss, metadata = run_grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=batch_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=batch_raw_rewards if loss_type == "no_baseline" else None,
                    advantages=batch_advantages if loss_type != "no_baseline" else None,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange if loss_type == "grpo_clip" else None,
                )

                # Step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    # Log training metrics
                    avg_entropy = run_masked_mean(token_entropy, batch_response_mask, dim=None).item()

                    wandb.log({
                        'train/loss': loss.item(),
                        'train/grad_norm': grad_norm.item(),
                        'train/entropy': avg_entropy,
                        'train/mean_reward': reward_metadata['mean'],
                        'train_step': train_step,
                    })

                    train_step += 1

        # === Evaluation Phase ===
        if (grpo_step + 1) % eval_steps == 0:
            print(f"\n[Evaluation Phase]")
            policy.eval()

            # Load current policy into vLLM
            load_policy_into_vllm_instance(policy, vllm_model)

            # Evaluate on validation set (sample for speed)
            eval_size = min(eval_batch_size, len(val_questions))
            eval_indices = random.sample(range(len(val_questions)), eval_size)
            eval_questions = [val_questions[i] for i in eval_indices]
            eval_ground_truths = [val_answers[i] for i in eval_indices]
            eval_prompts = format_prompts(eval_questions, prompt_template)

            eval_results = evaluate_vllm(
                vllm_model=vllm_model,
                reward_fn=r1_zero_reward_fn,
                prompts=eval_prompts,
                ground_truths=eval_ground_truths,
                eval_sampling_params=eval_sampling_params,
            )

            print(f"Validation Accuracy: {eval_results['accuracy']:.4f}")
            print(f"Validation Format Reward: {eval_results['format_reward']:.4f}")
            print(f"Validation Answer Reward: {eval_results['answer_reward']:.4f}")

            # Log some examples
            for i in range(min(3, len(eval_results['generated_texts']))):
                print(f"\nExample {i+1}:")
                print(f"Question: {eval_questions[i][:100]}...")
                print(f"Generated: {eval_results['generated_texts'][i][:200]}...")
                print(f"Ground Truth: {eval_ground_truths[i]}")

            wandb.log({
                'eval/accuracy': eval_results['accuracy'],
                'eval/format_reward': eval_results['format_reward'],
                'eval/answer_reward': eval_results['answer_reward'],
                'eval_step': eval_step,
            })

            eval_step += 1

        # Save checkpoint
        if (grpo_step + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{grpo_step+1}')
            print(f"\nSaving checkpoint to {checkpoint_path}")
            policy.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    # Final save
    final_path = os.path.join(save_dir, 'final_model')
    print(f"\nSaving final model to {final_path}")
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    wandb.finish()
    print("\nTraining complete!")

if __name__ == "__main__":
    app()