from pickletools import optimize
import timeit
from tracemalloc import start
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cs336-basics'))
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
import argparse
from typing import Tuple, Optional
import json
from pathlib import Path
from contextlib import nullcontext
model_configs = [
    # {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    # {"size": "medium","d_model": 1024,"d_ff": 4096,"num_layers": 24,"num_heads": 16,},
    # {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

#Hyperarameters
vocab_size = 10_000
context_length = 256
batch_size = 1
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# 放在文件里任意函数区（例如 benchmark() 上方或下方）
def profile_one_train_step(model, x, y, vocab_size, tag: str = ""):
    """
    仅执行一次训练步，并分别对 forward / backward / optimizer step
    开启独立的内存历史记录与导出快照（含 C++ + Python 栈）。
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # ---------- Forward ----------
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000, context='all', stacks='all'
    )
    torch.cuda.reset_peak_memory_stats()
    out = model(x)                       # 注意：不加 no_grad，保留激活供 backward
    torch.cuda.synchronize()
    fwd_alloc = torch.cuda.max_memory_allocated()
    fwd_res   = torch.cuda.max_memory_reserved()
    torch.cuda.memory._dump_snapshot(f"memory_forward{('_' + tag) if tag else ''}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    # ---------- Backward ----------
    loss = criterion(out.view(-1, vocab_size), y.view(-1))
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000, context='all', stacks='all'
    )
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    bwd_alloc = torch.cuda.max_memory_allocated()
    bwd_res   = torch.cuda.max_memory_reserved()
    torch.cuda.memory._dump_snapshot(f"memory_backward{('_' + tag) if tag else ''}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    del loss, out
    torch.cuda.synchronize()
    # ---------- Optimizer step ----------
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000, context='all', stacks='all'
    )
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()                     # 注意：第一次 step 会分配优化器状态
    torch.cuda.synchronize()
    opt_alloc = torch.cuda.max_memory_allocated()
    opt_res   = torch.cuda.max_memory_reserved()
    torch.cuda.memory._dump_snapshot(f"memory_optim{('_' + tag) if tag else ''}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    # 方便在控制台也看到峰值（字节）
    print(
        f"[PEAK] forward  alloc={fwd_alloc/1e9:.3f} GB, reserved={fwd_res/1e9:.3f} GB\n"
        f"[PEAK] backward alloc={bwd_alloc/1e9:.3f} GB, reserved={bwd_res/1e9:.3f} GB\n"
        f"[PEAK] optim    alloc={opt_alloc/1e9:.3f} GB, reserved={opt_res/1e9:.3f} GB"
    )



def benchmark(model, x, y, mode):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    def only_forward():
        with torch.no_grad():
            _ = model(x)
    def forward_and_backward():
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    step = only_forward if mode == 'forward' else forward_and_backward
    for _ in range(warmup_steps):
        step()
    times = []
    for _ in range(timing_steps):
        start_time = timeit.default_timer()
        step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time-start_time)
    return mean(times), stdev(times)

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Transofrmer models')
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_ff", type=int, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--all", action="store_true", help="Run all predefined configurations")
    parser.add_argument("--context_length", type=int, help="Sequence context length")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint", type=str, default="benchmark_checkpoint.json", help="Checkpoint file path")
    parser.add_argument('--mixed_percision', type=bool, default=False, help='Use mixed percision to run the model')
    return parser.parse_args()

def main():
    args = parse_args()
    global context_length
    if args.context_length:
        context_length = args.context_length
    global warmup_steps
    if args.warmup_steps:
        warmup_steps = args.warmup_steps
    use_mixed = args.mixed_percision
    # Load checkpoint if resume is enabled
    checkpoint_path = Path(args.checkpoint)
    res = []
    completed_tasks = set()

    if args.resume and checkpoint_path.exists():
        print(f"\n[Resume] Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            res = checkpoint_data.get('results', [])
            completed_tasks = set(tuple(sorted(task.items())) for task in checkpoint_data.get('completed_tasks', []))
        print(f"[Resume] Found {len(res)} completed benchmarks")

    run_configs = []
    if args.all:
        run_configs = model_configs
    elif args.d_model and args.d_ff and args.num_layers and args.num_heads:
        run_configs = [{
            "size": "custom",
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }]
    else:
        raise ValueError("Must specify either --all or all custom model hyperparameters.")
    print("\nRunning the following configurations:")
    for cfg in run_configs:
        print(f"  - {cfg}")
    print()
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if use_mixed else nullcontext()
    with ctx:
        for config in run_configs:
            for mode in ['forward_and_backward']:#for mode in ['forward', 'forward_and_backward']
                # Create task identifier
                task_id = {
                    "size": config["size"],
                    "mode": mode,
                    "d_model": config["d_model"],
                    "d_ff": config["d_ff"],
                    "num_layers": config["num_layers"],
                    "num_heads": config["num_heads"]
                }
                task_tuple = tuple(sorted(task_id.items()))

                # Skip if already completed
                if task_tuple in completed_tasks:
                    print(f"  [Skip] {config['size']} [{mode}] - already completed")
                    continue

                print(f"\n[Running] {config['size']} [{mode}]...")
                model = BasicsTransformerLM(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    d_ff=config["d_ff"],
                    rope_theta=rope_theta
                ).to(device)
                x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
                y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
                tag = f"{config['size']}_ctx{context_length}_{'bf16' if use_mixed else 'fp32'}"
                profile_one_train_step(model, x, y, vocab_size, tag=tag)
                # time_mean, time_std = benchmark(model, x, y, mode)
                # print(f"  [Done] {config['size']} [{mode}]: Avg Time = {time_mean:.6f}s, Std Dev = {time_std:.6f}s")
                del model, x, y
                torch.cuda.empty_cache()

                # result = {
                #     "Size": config["size"],
                #     "Mode": mode,
                #     "d_model": config["d_model"],
                #     "d_ff": config["d_ff"],
                #     "num_layers": config["num_layers"],
                #     "num_heads": config["num_heads"],
                #     "Context Length": context_length,
                #     "Avg Time (s)": round(time_mean, 6),
                #     "Std Dev (s)": round(time_std, 6),
                #     "Warmup Steps": warmup_steps,
                # }
                # res.append(result)
                # completed_tasks.add(task_tuple)

                # # Save checkpoint after each completed benchmark
                # checkpoint_data = {
                #     'results': res,
                #     'completed_tasks': [dict(task) for task in completed_tasks]
                # }
                # with open(checkpoint_path, 'w') as f:
                #     json.dump(checkpoint_data, f, indent=2)
                # print(f"  [Checkpoint] Saved to {checkpoint_path}")
    
    res = pd.DataFrame(res)
    print(res.to_markdown())
    with open('benchmark_res(1).md', 'w') as f:
        f.write(res.to_markdown())
    
if __name__=='__main__':
    main()