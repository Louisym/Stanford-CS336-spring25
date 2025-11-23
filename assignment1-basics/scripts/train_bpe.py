import os
import sys
import pickle
import pathlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import run_train_bpe



DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / 'data'
INPUT_PATH = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-train.txt')

#保存路径
TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

#config
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

#train
vocab, merge = run_train_bpe(
    input_path=INPUT_PATH,
    vocab_size=vocab_size,
    special_tokens=special_tokens
)

os.makedirs(TOKENIZER_DIR, exist_ok=True)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, "wb") as f:
    pickle.dump(merge, f)

longest_token = max(vocab.values(), key=len)
print("最长token:", longest_token, "长度:", len(longest_token))