import os
# Workaround for WSL LocalFileSystem issue - use native Linux path for cache
os.environ['HF_DATASETS_CACHE'] = '/tmp/huggingface_cache'

from datasets import load_dataset

# Load dataset directly from parquet files on HuggingFace
print("Downloading Countdown-Tasks-3to4 dataset...")
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

print(f"\nDataset downloaded successfully!")
print(f"Dataset info: {ds}")
print(f"Number of examples: {len(ds['train'])}")
print(f"\nFirst example:")
print(ds['train'][0])