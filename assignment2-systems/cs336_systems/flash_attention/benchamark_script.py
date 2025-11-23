import time
import timeit
import torch
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.model import scaled_dot_product_attention
import argparse
from tqdm import tqdm


batch_size = 8
num_heads = 1
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192]
loop = 100
warm_up = 5
@torch.compile
def pytorch_attention(d_model, seq_len, device):
    #randomly and dicrectly create Q K V
    
    f_times, b_times = [], []
    fwd_peak_mb = None
    try:
        Q = torch.randn(batch_size,  seq_len,d_model, dtype=torch.float32, device=device, requires_grad=True)
        K = torch.randn(batch_size,  seq_len,d_model, dtype=torch.float32, device=device, requires_grad=True)
        V = torch.randn(batch_size,  seq_len,d_model, dtype=torch.float32, device=device, requires_grad=True)
        #warm up process
        for _ in tqdm(range(warm_up), desc=f'[d={d_model}, seq={seq_len}] Warmup', leave=False):
            _ = scaled_dot_product_attention(Q,K,V)
        torch.cuda.reset_peak_memory_stats()
        #forward process
        for _ in tqdm(range(loop), desc=f'[d={d_model}, seq={seq_len}] Forward', leave=False):
            torch.cuda.synchronize()
            start = timeit.default_timer()
            attn = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            f_times.append(end - start)
            Q.grad = K.grad = V.grad = None
        fwd_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        #backward process
        for _ in tqdm(range(loop), desc=f'[d={d_model}, seq={seq_len}] Backward', leave=False):
            attn = scaled_dot_product_attention(Q, K, V)
            loss = attn.sum()
            torch.cuda.synchronize()
            back_start = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            back_end = timeit.default_timer()
            b_times.append(back_end - back_start)
            Q.grad, K.grad, V.grad = None, None, None
        del Q, K, V
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    result = {
        "seq_len": seq_len,
        "d_model": d_model,
        "forward_ms_mean": None if not f_times else round(mean(f_times) * 1e3, 3),
        "forward_ms_std":  None if not f_times else round(stdev(f_times) * 1e3, 3),
        "backward_ms_mean": None if not b_times else round(mean(b_times) * 1e3, 3),
        "backward_ms_std":  None if not b_times else round(stdev(b_times) * 1e3, 3),
        "fwd_peak_mem_MB":  None if fwd_peak_mb is None else round(fwd_peak_mb, 1),
        # "status": status,
        # "mixed": ("bf16" if (use_mixed and mixed_dtype==torch.bfloat16) else
        #           "fp16" if (use_mixed and mixed_dtype==torch.float16) else "fp32"),
        "batch_size": batch_size,
    }
    return result

 
def main():
    res = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', 'No GPU available!'
    for d_model in d_models:
        for seq_len in seq_lens:
            result = pytorch_attention(d_model, seq_len, device)
            res.append(result)
    res = pd.DataFrame(res)
    with open('attn_time_benchmark_opt.md', 'w') as f:
        f.write(res.to_markdown())
    
if __name__ == '__main__':
    main()
    


