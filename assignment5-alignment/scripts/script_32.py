"""
Script for Problem 3.2: Zero-shot Math Baseline Evaluation

This script evaluates Qwen 2.5 Math 1.5B zero-shot performance on math datasets (MATH/GSM8K).
Uses standardized 'question'/'answer' format for all datasets.
"""

import json
import os
from pathlib import Path
from typing import List, Callable, Dict
import argparse

from vllm import LLM, SamplingParams
from tqdm import tqdm

# Import the reward function
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_r1_zero_prompt() -> str:
    """Load the r1_zero prompt template."""
    prompt_path = Path("cs336_alignment/prompts/r1_zero.prompt")
    with open(prompt_path, "r") as f:
        return f.read()


def extract_gsm8k_answer(answer: str) -> str:
    """Extract final numerical answer from GSM8K format.

    GSM8K answers end with #### <number>
    Example: "Some reasoning steps...\n#### 18" -> "18"
    """
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def format_prompts(examples: List[Dict], prompt_template: str) -> List[str]:
    """Format math examples using the r1_zero prompt template.

    Args:
        examples: List of examples with 'question' field
        prompt_template: The prompt template with {question} placeholder

    Returns:
        List of formatted prompts
    """
    prompts = []
    for example in examples:
        # Get question (with backward compatibility for old 'problem' key)
        question = example.get("question") or example.get("problem")
        if question is None:
            raise ValueError(f"Example missing 'question' field: {example}")
        # The prompt template has {question} placeholder
        prompt = prompt_template.replace("{question}", question)
        prompts.append(prompt)
    return prompts


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str = None,
) -> Dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    Args:
        vllm_model: vLLM model instance
        reward_fn: Function that takes (response, ground_truth) and returns dict with reward scores
        prompts: List of formatted prompts
        ground_truths: List of ground truth answers
        eval_sampling_params: Sampling parameters for generation
        output_path: Optional path to save results

    Returns:
        Dictionary with evaluation metrics and results
    """
    print(f"Generating responses for {len(prompts)} prompts...")

    # Generate outputs
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Extract generated text
    generated_texts = [output.outputs[0].text for output in outputs]

    # Evaluate each generation
    print("Evaluating generations...")
    results = []

    for prompt, generated_text, ground_truth in tqdm(
        zip(prompts, generated_texts, ground_truths),
        total=len(prompts),
        desc="Evaluating"
    ):
        # Compute rewards
        reward_dict = reward_fn(generated_text, ground_truth)

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "reward": reward_dict["reward"],
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
        })

    # Compute aggregate statistics
    total_examples = len(results)

    # Category 1: Both format and answer reward = 1
    correct_both = sum(1 for r in results if r["format_reward"] == 1 and r["answer_reward"] == 1)

    # Category 2: Format reward = 1, answer reward = 0
    correct_format_wrong_answer = sum(1 for r in results if r["format_reward"] == 1 and r["answer_reward"] == 0)

    # Category 3: Format reward = 0
    wrong_format = sum(1 for r in results if r["format_reward"] == 0)

    # Overall metrics
    avg_total_reward = sum(r["reward"] for r in results) / total_examples
    avg_format_reward = sum(r["format_reward"] for r in results) / total_examples
    avg_answer_reward = sum(r["answer_reward"] for r in results) / total_examples

    metrics = {
        "total_examples": total_examples,
        "correct_both": correct_both,
        "correct_format_wrong_answer": correct_format_wrong_answer,
        "wrong_format": wrong_format,
        "avg_total_reward": avg_total_reward,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
        "accuracy": avg_answer_reward,  # This is the key metric
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total examples: {total_examples}")
    print(f"\nCategory breakdown:")
    print(f"  (1) Correct format + Correct answer: {correct_both} ({100*correct_both/total_examples:.1f}%)")
    print(f"  (2) Correct format + Wrong answer: {correct_format_wrong_answer} ({100*correct_format_wrong_answer/total_examples:.1f}%)")
    print(f"  (3) Wrong format: {wrong_format} ({100*wrong_format/total_examples:.1f}%)")
    print(f"\nAverage rewards:")
    print(f"  Total reward: {avg_total_reward:.4f}")
    print(f"  Format reward: {avg_format_reward:.4f}")
    print(f"  Answer reward (accuracy): {avg_answer_reward:.4f}")
    print("="*60 + "\n")

    # Save results if output path provided
    if output_path:
        output_data = {
            "metrics": metrics,
            "results": results,
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot MATH baseline")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/c/Users/louis/louis-tmp/assignment5-alignment/Qwen2.5-Math-1.5B",
        help="Path to model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/math/test.jsonl",
        help="Path to math dataset (JSONL format)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/math_baseline_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum generation length"
    )

    args = parser.parse_args()

    # Load MATH validation data
    print(f"Loading data from {args.data_path}...")
    examples = []
    with open(args.data_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Load prompt template
    prompt_template = load_r1_zero_prompt()

    # Format prompts
    print("Formatting prompts...")
    prompts = format_prompts(examples, prompt_template)

    # Extract ground truths - use 'answer' field (with backward compatibility for 'solution')
    ground_truths = []
    for ex in examples:
        gt = ex.get("answer") or ex.get("solution")
        if gt is None:
            raise ValueError(f"Example missing 'answer' field: {ex}")
        # For GSM8K, extract final answer from #### format
        if "####" in gt:
            gt = extract_gsm8k_answer(gt)
        ground_truths.append(gt)

    # Initialize vLLM model
    print(f"Loading model from {args.model_path}...")
    llm = LLM(model=args.model_path)

    # Set up sampling parameters
    # Stop at </answer> tag as specified in the PDF
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Run evaluation
    metrics, results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=args.output_path,
    )

    # Print some example outputs for analysis
    print("\n" + "="*60)
    print("SAMPLE OUTPUTS (first 3)")
    print("="*60)
    for i in range(min(3, len(results))):
        print(f"\n--- Example {i+1} ---")
        # Use flexible key access for compatibility
        question = examples[i].get('question') or examples[i].get('problem', 'N/A')
        print(f"Question: {question[:100]}...")
        print(f"Generated: {results[i]['generated_text'][:200]}...")
        print(f"Ground truth: {results[i]['ground_truth']}")
        print(f"Rewards - Format: {results[i]['format_reward']}, Answer: {results[i]['answer_reward']}")
    print("="*60)


if __name__ == "__main__":
    main()
