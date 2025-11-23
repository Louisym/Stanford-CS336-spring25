import gc
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
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
from sft_experiment import *
from cs336_alignment import *

logging.getLogger('vllm').setLevel(logging.WARNING)

@dataclass
class Trainconfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "ei-qwen2.5"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    num_example: int = 128

    #training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    training_steps: int = 8
    mixed_precision_training: bool = True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    #eval
    log_print_steps: int = 12
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 12

    #ei
    n_ei_steps: int = 4
    num_questions: int = 256
    G_responses: int = 8

    #Learning Rate adjust
    lr_mode: str = "global"  # one of: "global", "per_outer", "constant"
    warmup_ratio: float = 0.03  # 0 = no warmup
    min_lr_factor: float = 0.1  # min_lr = lr * min_lr_factor
    scale_lr_by_pairs: bool = True  # scale lr by 1/sqrt(num_kept_pairs) per EI step

    #ei vllm generation setup
    ei_temperature: float = 1.0
    ei_top_p: float = 1.0
    ei_max_tokens: int = 1024
    ei_stop_tokens: list[str] = field(default_factory=lambda: ['</answer>'])
    ei_include_stop_token: bool = True
    ei_min_tokens: int = 4

@dataclass
class Evalconfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])
    max_tokens: int = 1024
    include_stop_str_in_output: bool = True


def cycle_dataloader(dataloader):
    """
    Creates a cycling iterator for a PyTorch DataLoader.
    """
    while True:
        for batch in dataloader:
            yield batch

#we should slightly change the sft_experiment function
def train_sft_ei(
        HF_model,
        tokenizer,
        train_config: Trainconfig,
        eval_config: Evalconfig,
        ei_prompts,
        ei_cot,
        ei_answers,
        vllm: LLM,
        evaluate: bool = True,
        ei_step: int,
        global_steps: int ,
):
    dataset = SFTDataset(ei_prompts, ei_cot, ei_answers)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: sft_rewrite_response(batch, tokenizer)
    )

    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if train_config.mixed_precision_training
        else nullcontext()
    )

    optimizer = torch.optim.AdamW(HF_model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)

    cur_step = 0
    batch_loss = 0
    loss = 0
    total_micro_steps = 0
    global_step_ = global_steps
    while True:
        for b_i, data in enumerate(dataloader):
            total_micro_steps += 1
            input_ids = data['input_ids'].to(device=train_config.train_device)
            labels = data['labels'].to(device=train_config.train_device)
            response_mask = data['response_mask'].to(device=train_config.train_device)

            with ctx:
                log_prob = run_get_response_log_probs(HF_model, input_ids, labels, False)
                log_prob = log_prob['log_probs']
                l, _ = run_sft_microbatch_train_step(log_prob, response_mask, train_config.gradient_accumulation_steps)
            
            batch_loss += l

            if total_micro_steps % train_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(HF_model.parameters(), max_norm=1.0)
                #Note!! we dont impelement global_steps here. As every EI dataset is a brand new dataset. There is no connective schedule intuiatively here.
                adj_lr = get_lr(ei_step, train_config.n_ei_steps, train_config.learning_rate)
                for param in optimizer.param_groups:
                    param['lr'] = adj_lr
                
                optimizer.step()
                optimizer.zero_grad()
                loss += batch_loss / train_config.gradient_accumulation_steps

                print(
                    f"[train | ei step-{ei_step}] Step {global_step_} | Loss: {batch_loss / train_config.gradient_accumulation_steps:.4f} | LR: {adj_lr:.6f}"
                )
                wandb.log(
                    {
                        "train/loss": batch_loss / train_config.gradient_accumulation_steps,
                        "train/lr": adj_lr,
                        "train/step": global_step_,
                    }
                )

                batch_loss = 0
                cur_step += 1
                global_step_ += 1

        if cur_step >= train_config.training_steps:
            break
    return loss / cur_step, global_step_

class EIDataset(Dataset):
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
    
def generate_and_select_outputs_G(
        vllm: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        answers: List[str],
        train_config: Trainconfig,
) -> tuple[List[str], List[str]]:
    G_prompts = []
    G_outputs = []
    ei_sampling_params = SamplingParams(
        temperature=train_config.ei_temperature,
        top_p=train_config.ei_top_p,
        max_tokens=train_config.ei_max_tokens,
        stop=train_config.ei_stop_tokens,
        include_stop_str_in_output=train_config.ei_include_stop_token,
        min_tokens=train_config.ei_min_tokens,
        n=train_config.G_responses
    )

    all_generations = vllm.generate(prompts, ei_sampling_params)
    for p, ans, gs_per_q in zip(prompts, answers, all_generations):
        #we will select all the output that reward score matches.
        for _, g in enumerate(gs_per_q.outputs):
            score = reward_fn(g.text, ans)
            if score['reward'] == 1:
                G_prompts.append(p)
                G_outputs.append(g.text)
    return (G_prompts, G_outputs)
    
def train_ei_experiment(
    train_config: Trainconfig,
    eval_config: Evalconfig,
    train_prompts,
    train_cot,
    train_answers,
    vllm: LLM,
):
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-ei",
        config={"train": asdict(train_config), "eval": asdict(eval_config)},
        name='ei',
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    eval_samplingparams = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=train_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(device=train_config.train_device)

    tokenizer =AutoTokenizer.from_pretrained(
        train_config.model_name,
    )

    optimizer = torch.optim.AdamW(
        hf_model.parameters(),
        train_config.learning_rate,
        train_config.betas,
    )

    print(f"[ei train] Tokenizer {train_config.model_name} loaded")
    print(f"[ei train] Model {train_config.model_name} loaded on {train_config.train_device}")
    print("[ei train] Optimizer loaded")

    #step1: sample questions from the raw dataset
    raw_dataset = EIDataset(
        train_prompts,
        train_cot,
        train_answers,
    )
    raw_dataloader = DataLoader(
        raw_dataset,
        train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    loader = cycle_dataloader(raw_dataloader)
    global_steps = 0
    for ei in range(train_config.n_ei_steps):
        batch = next(loader)
        prompts, cot, answers = batch[0], batch[1], batch[2]

        G_prompts, G_outputs = generate_and_select_outputs_G(
            vllm,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            answers=answers,
            train_config=train_config,
        )

        if len(G_prompts) == 0:
            print(f"[EI] Step {ei}: no correct generations; skipping SFT update.")
            continue

        print_color(f"[EI] Step {ei} | Pairs: {len(G_prompts)}")
        print("[EI] Example kept pair:")
        assert len(G_prompts) == len(G_outputs)
        print("Kept prompts: ", G_prompts[0])
        print("Kept outputs: ", G_outputs[0])

        #optimize the policy model using new expert dataset
        avg_loss, global_steps = train_sft_ei(
            hf_model,
            tokenizer,
            train_config,
            eval_config,
            G_prompts,
            G_outputs,
            [0]*len(G_prompts),
            vllm,
            evaluate=True,
            ei_step=ei,
            global_steps=global_steps,
        )
    
        print_color(f"[EI] Step {ei} | Pairs: {len(G_prompts)} | Loss: {avg_loss:.4f}", color="green")

        # wandb.log({"train/pairs": len(kept_prompts), "train_step": step, "train/loss": loss})

        print(f"Loaded weights to vllm at step {ei}")
        load_model_into_vllm_instance(hf_model, vllm)
        # Periodic qualitative logging and eval
        if (ei + 1) % train_config.log_print_steps == 0:
            log_generations(
                vllm,
                reward_fn=r1_zero_reward_fn,
                prompts=raw_dataset.train_prompts,
                cot=raw_dataset.train_cot,
                answers=[str(x) for x in raw_dataset.train_answers],
                eval_sampling_params=eval_samplingparams,
                cur_step=ei,
                num_example=2,
            )

        if (ei + 1) % train_config.eval_interval_steps == 0:
            eval_sft_experiment(eval_config, vllm, global_steps)
            save_model_and_tokenizer(hf_model, tokenizer, train_config)

    save_model_and_tokenizer(hf_model, tokenizer, train_config)
    wandb.finish()
    return hf_model

def main(
        model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/math/train.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int = 42,
):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/workspace/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf/models"
    os.environ["HF_HUB_CACHE"] = "/workspace/hf/hub"

    train_config = Trainconfig()
    eval_config = Evalconfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    vllm = init_vllm(model_id=model_name, device=train_config.eval_device, seed=seed)
    prompts, cot, answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    for num_samples in [len(prompts)]:
        train_config.num_example = num_samples
        train_config.experiment_name = f"experiment_{num_samples}"
        train_config.n_ei_steps = math.ceil(num_samples / train_config.question_per_step)

        print_rich_dict({"train config": asdict(train_config), "eval config": asdict(eval_config)})

        train_prompts = prompts[:num_samples]
        train_cot = cot[:num_samples]
        train_answers = answers[:num_samples]

        train_ei_experiment(
            train_config,
            eval_config=eval_config,
            train_prompts=train_prompts,
            train_cot=train_cot,
            train_answers=train_answers,
            vllm=vllm,
        )

    wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)