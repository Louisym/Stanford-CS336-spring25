import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List

import dotenv
import fire
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from tests.adapters import *

from cs336_alignment.utils.log_generations import log_generations

#config logger
logging.getLogger('vllm').setLevel(logging.WARNING)

#config the trainer
@dataclass
class TrainConfig:
    experiment_name: str = 'sft-qwen2.5'
    model_name: str = 'Qwen2.5-Math-1.5B'
    data_path: str = '/mnt/c/Users/louis/louis-tmp/assignment5-alignment/data/math/train.jsonl'
    prompt_path: str = '/mnt/c/Users/louis/louis-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'

    #train config
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    training_steps: int = 128
    mixed_percision_training: bool = True
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.90, 0.98)
    train_device: str = 'cuda:0'

    num_examples: int = 128

    log_print_steps = 12
    
    #eval config
    eval_device: str = 'cuda:1'
    eval_interval_steps: int = 32

@dataclass
class EvaluateConfig:
    data_path: str = '/mnt/c/Users/louis/louis-tmp/assignment5-alignment/data/math/test.jsonl'
    prompt_path: str = '/mnt/c/Users/louis/louis-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024

def sft_rewrite_response(batch, tokenizer):
    prompt, CoT, answer = zip(*batch)
    #tokenizer receives list inputs
    prompt = list(prompt)
    CoT = list(CoT)

    tokenized_info = run_tokenize_prompt_and_output(prompt, CoT, tokenizer)

    return {
        **tokenized_info,
        'answer': answer
    }

#standardize the data type
def transfer_to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float().item()
    elif isinstance(x, str):
        return float(x.strip())
    return float(x)

class SFTDataset(Dataset):
    def __init__(self, prompt, cot, answer):
        self.prompt = prompt
        self.cot = cot
        self.answer = answer
    def __len__(self):
        return len(self.prompt)
    def __getitem__(self, index):
        prom = self.prompt[index]
        cot = self.cot[index]
        ans = self.answer[index]
        return (prom, cot, ans)

def get_lr(it, max_steps, max_lr):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    ratio = (it / max_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)

def get_response(vllm, params, prompts):
    result = vllm.generate(prompts, params)
    outputs = [out.output[0].text.strip() for out in result]
    return outputs

def extract_reference_answer(response: str) -> str:
    from cs336_alignment.drgrpo_grader import extract_answer

    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

        return model_answer
def print_rich_dict(data: dict) -> None:
    """Pretty print dictionary with colors using rich."""
    from rich.pretty import pprint

    pprint(data, expand_all=True)


def generate(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    cot: List[str],
    answer: List[str],
    eval_params: SamplingParams,
    cur_step: int,
    num_examples: int = 2,
):
    random_samples = random.sample(range(len(prompts)), num_examples)
    sampled_prom = [prompts[i] for i in random_samples]
    sampled_cot = [cot[i] for i in random_samples]
    sampled_ans =  [answer[i] for i in random_samples]

    

    responses = get_response(vllm_model, eval_params, sampled_prom)


    for i in range(num_examples):
        prompt = sampled_prom[i]
        cot = sampled_cot[i]
        ans = sampled_ans[i]
        response = responses[i]
        pure_ans = extract_reference_answer(response)

        reward_dict = reward_fn(response, ans)

        info = {
            'prompt': prompt,
            'cot': cot,
            'answer': ans,
            'vllm_response': response,
            'vllm_answer': pure_ans,
            **reward_dict
        }
        print(f"======= Step: {cur_step}; Example {i} =======")
        # print_formatted_dict(info_dict)
        print_rich_dict(info)
        print("==============================================\n")

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
):
    '''
    this function is to overview the avaluation results
    '''
    import collections
    responses = get_response(vllm_model, eval_sampling_params, prompts)
    
    reward_dicts = []
    for p, a, r in zip(prompts, answers, responses):
        reward_dict = reward_fn(r, a)
        reward_dicts.append(reward_dict)
    
    all_res = collections.defaultdict(int)
    for rw_dict in reward_dicts:
        for res, times in rw_dict.items():
            all_res['count'] += 1
            all_res[res] += times
    return all_res

import json
def convert_cot_to_think_answer(text: str) -> str:
    """
    Convert a chain-of-thought style answer that ends with a line like
    "#### 5" into the desired format by replacing that trailer with
    " </think> <answer> 5 </answer>".

    Examples
    --------
    >>> s = (
    ...     "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\\n"
    ...     "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\\n"
    ...     "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\\n"
    ...     "#### 5"
    ... )
    >>> convert_cot_to_think_answer(s)
    "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. </think> <answer> 5 </answer>"

    If no trailing "#### <ans>" is found, this function will try to extract a
    terminal number at the end of the string and use that as the answer. If that
    also fails, the input text is returned unchanged.
    """
    # Match a final line that looks like: #### 5 (possibly with spaces/newline)
    m = re.search(r"####\s*([^\n]+)\s*$", text)
    if m:
        ans = m.group(1).strip()
        prefix = text[: m.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    # Fallback: try to capture a trailing number at end of text
    m_num = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    if m_num:
        ans = m_num.group(1)
        prefix = text[: m_num.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    return text
import re
def extract_math_answer(answer: str) -> str:
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def load_and_format_prompts(data_path: str, prompt_path: str) -> tuple[list[str], list[str], list[str]]:
    with open(prompt_path, "r") as file:
        prompt = file.read()

    prompts = []
    cot = []
    answers = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))
            cot.append(convert_cot_to_think_answer(data["answer"]))
            answers.append(extract_math_answer(data["answer"]))

    return prompts, cot, answers

from cs336_alignment.drgrpo_grader import *
def eval_sft_experiment(
        config: EvaluateConfig, vllm: LLM, eval_step: int
):
    prompts, cot, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    #define samping configs here
    samplingparams = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=['</answer>'],
        include_stop_str_in_output=True,
    )

    eval_res = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, samplingparams)
    wandb.log(
        {
            **eval_res,
            'eval_step': eval_step
        }
        
    )

def load_model_into_vllm_instance(model: torch.nn.Module, llm: LLM):
    # snapshot to CPU -> then load into vLLM
    model.eval()
    model.tie_weights()
    cpu_state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cpu_state_dict.items())
    model.train()
    torch.cuda.synchronize(torch.device("cuda:1"))
    print("Model weights loaded into VLLM instance.")

from pathlib import Path
def save_model_and_tokenizer(model, tokenizer, config):
    out_dir = Path(f"./{config.experiment_name_base}/{config.experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Model and tokenizer saved to {out_dir}")


def train_sft_experiment(
        HF_model,
        tokenizer,
        train_config: TrainConfig,
        eval_config: EvaluateConfig,
        train_prompts,
        train_cot, 
        train_answers,
        vllm: LLM,
        evaluate_mode: bool = True,
):
    wandb.init(
        entity=os.getenv('WANDB_ENTITY'),
        project='cs336_sft_experiment',
        config={
            'train_config': asdict(train_config),
            'eval_config': asdict(eval_config),
        },
        name='sft',
        reinit=True,
    )
    wandb.define_metric('train_step')
    wandb.define_metric('eval_step')
    wandb.define_metric('train/*', step_metric='train_step')
    wandb.define_metric('eval/*', step_metric='eval_step')

    sft_dataset = SFTDataset(train_prompts, train_cot, train_answers)
    data_loader = DataLoader(sft_dataset, train_config.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=lambda batch: sft_rewrite_response(batch, tokenizer))
    print(f"Dataloader has been initialized successfully with batch size {train_config.batch_size}")

    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if train_config.mixed_percision_training else nullcontext()

    optimizer = torch.optim.AdamW(HF_model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
    print(f'Optimizer has been initialized successfully')

    #now preparation is done, we start training loop
    cur_step = 0
    loss = 0.0
    micro_steps = 0
    while True:
        for i, train_data in enumerate(data_loader):
            micro_steps += 1
            input_ids = train_data['input_ids'].to(device=train_config.train_device)
            labels = train_data['labels'].to(device=train_config.train_device)
            response_mask = train_data['response_mask'].to(device=train_config.train_device)
            answers = train_data['answers'].to(device=train_config.train_device)

            with ctx:
                #cal the loss
                log_prob_info = run_get_response_log_probs(HF_model, input_ids, labels, return_token_entropy=True)
                log_prob = log_prob_info['log_probs']
                microbatch_loss, l_info = run_sft_microbatch_train_step(log_prob, response_mask, train_config.gradient_accumulation_steps)

            loss += microbatch_loss

            #if it arrives the accumulation steps we set, optimize it
            if micro_steps % train_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(HF_model.parameters(), max_norm=1.0)
                lr = get_lr(cur_step, train_config.training_steps, train_config.learning_rate)
                for param in optimizer.param_groups:
                    param['lr'] = lr
                
                optimizer.step()
                optimizer.zero_grad()

                print(
                    f"[train] Step {cur_step} | Loss: {loss / train_config.gradient_accumulation_steps:.4f} | LR: {lr:.6f}"
                )
                wandb.log(
                    {
                        'train/loss': loss / train_config.gradient_accumulation_steps,
                        'train/lr': lr,
                        'train/step': cur_step,
                    }
                )

                loss = 0
                cur_step += 1

                if (cur_step + 1) % train_config.log_print_steps == 0 and evaluate_mode:
                    load_model_into_vllm_instance(HF_model, vllm)
                    generate(
                        vllm, r1_zero_reward_fn, sft_dataset.prompt, sft_dataset.cot, sft_dataset.answer, SamplingParams(
                            temperature=eval_config.temperature,
                            top_p=eval_config.top_p,
                            max_tokens=eval_config.max_tokens,
                            stop=["</answer>"],
                            include_stop_str_in_output=True,
                        ), cur_step,
                    )

                if (cur_step + 1) % train_config.eval_interval_steps == 0 and evaluate_mode:
                    print(f'saving model with the {cur_step} step\'s state')
                    save_model_and_tokenizer(HF_model, tokenizer, train_config)
                    print(f"[eval] at step {cur_step}")
                    load_model_into_vllm_instance(HF_model, vllm)
                    eval_sft_experiment(eval_config, vllm, cur_step)
                    print(f"[eval] Evaluation completed for step {cur_step}")

            if cur_step >= train_config.training_steps:
                break
    
    save_model_and_tokenizer(model, tokenizer, train_config)
    print(f"[train] Training finished at step {cur_step}")

    wandb.finish()
    return HF_model

from unittest.mock import patch
from vllm.model_executor import set_random_seed
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def main(
        model_name: str = 'Qwen/Qwen2.5-Math-1.5B',
        data_path: str = '/mnt/c/Users/louis/louis-tmp/assignment5-alignment/data/math/train.jsonl',
        prompt_path: str = '/mnt/c/Users/louis/louis-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt',
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        seed: int = 42,
):
    dotenv.load_dotenv()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    train_config = TrainConfig()
    eval_config = EvaluateConfig()

    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)

    vllm = init_vllm(model_id=model_name, device=train_config.eval_device, seed=seed)

    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    iterations = [len(prompts)]#to be extended
    for num_samples in iterations:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        ).to(train_config.train_device)
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

        train_config.num_examples = num_samples

        train_prompts = prompts[:num_samples]
        train_cot = cot[:num_samples]
        train_answers = answers[:num_samples]

        train_sft_experiment(
            model, tokenizer, train_config, eval_config, train_prompts, train_cot, train_answers, vllm, 
        )

    wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
    