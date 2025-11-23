import torch
import wandb
from transformers import PreTrainedTokenizerBase


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    labels: list[str] | None = None,
    num_samples: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    step: int | None = None,
):
    """Log model generations to Weights & Biases for monitoring training progress.

    Args:
        model: The model to generate with
        tokenizer: The tokenizer to use for encoding/decoding
        prompts: List of prompt strings to generate from
        labels: Optional list of ground truth labels/responses
        num_samples: Number of prompts to sample and log
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        step: Optional training step number for logging
    """
    if not wandb.run:
        # WandB is not initialized, skip logging
        return

    model.eval()
    device = next(model.parameters()).device

    # Sample a subset of prompts
    num_to_log = min(num_samples, len(prompts))
    indices = torch.randperm(len(prompts))[:num_to_log].tolist()

    table_data = []

    with torch.no_grad():
        for idx in indices:
            prompt = prompts[idx]

            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            # Decode the generation
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part (remove prompt)
            generated_response = generated_text[len(prompt):].strip()

            # Prepare row data
            row = [prompt, generated_response]
            if labels is not None and idx < len(labels):
                row.append(labels[idx])

            table_data.append(row)

    # Create WandB table
    columns = ["Prompt", "Generated"]
    if labels is not None:
        columns.append("Ground Truth")

    table = wandb.Table(columns=columns, data=table_data)

    # Log to WandB
    log_key = "generations" if step is None else f"generations_step_{step}"
    wandb.log({log_key: table}, step=step)

    model.train()