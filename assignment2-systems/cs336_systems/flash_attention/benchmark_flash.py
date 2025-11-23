"""
Benchmarking script for FlashAttention-2 vs PyTorch attention
"""
import torch
import triton.testing
import pandas as pd
from cs336_systems.flash_attention.flash_att_triton import flash_attention_triton
import itertools


def benchmark_attention(impl_name, attention_fn, seq_len, d_model, dtype, batch_size=1, is_causal=True, device='cuda'):
    """Benchmark a single attention implementation"""
    # Create random inputs
    q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)

    # Warmup
    for _ in range(5):
        o = attention_fn(q, k, v, is_causal)
        if o.requires_grad:
            grad_out = torch.randn_like(o)
            o.backward(grad_out)
            q.grad = None
            k.grad = None
            v.grad = None

    # Benchmark forward
    def forward_fn():
        return attention_fn(q, k, v, is_causal)

    forward_time = triton.testing.do_bench(forward_fn)

    # Benchmark backward
    def backward_fn():
        o = attention_fn(q, k, v, is_causal)
        grad_out = torch.randn_like(o)
        o.backward(grad_out)
        q.grad = None
        k.grad = None
        v.grad = None

    backward_time = triton.testing.do_bench(backward_fn)

    # End-to-end time is approximately forward + backward
    # But let's measure it directly
    end_to_end_time = triton.testing.do_bench(backward_fn)

    return {
        'implementation': impl_name,
        'seq_len': seq_len,
        'd_model': d_model,
        'dtype': str(dtype),
        'forward_ms': forward_time,
        'backward_ms': backward_time,
        'end_to_end_ms': end_to_end_time
    }


def pytorch_attention(q, k, v, is_causal):
    """Reference PyTorch attention implementation"""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = q @ k.transpose(-2, -1) * scale

    if is_causal:
        seq_len = q.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out


def main():
    # Test configurations - reduced for faster benchmarking
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192]
    d_models = [64, 128]
    dtypes = [torch.bfloat16]

    results = []

    print("Starting benchmarking...")
    print(f"Testing {len(seq_lengths)} sequence lengths, {len(d_models)} embedding dims, {len(dtypes)} dtypes")
    print(f"Total configurations: {len(seq_lengths) * len(d_models) * len(dtypes) * 2}")  # *2 for flash and pytorch

    for seq_len, d_model, dtype in itertools.product(seq_lengths, d_models, dtypes):
        print(f"\nTesting seq_len={seq_len}, d_model={d_model}, dtype={dtype}")

        try:
            # Benchmark PyTorch attention
            print("  - PyTorch attention...")
            pytorch_result = benchmark_attention(
                'PyTorch', pytorch_attention, seq_len, d_model, dtype
            )
            results.append(pytorch_result)
            print(f"    Forward: {pytorch_result['forward_ms']:.3f}ms, Backward: {pytorch_result['backward_ms']:.3f}ms")
        except RuntimeError as e:
            print(f"  - PyTorch attention OOM: {e}")
            results.append({
                'implementation': 'PyTorch',
                'seq_len': seq_len,
                'd_model': d_model,
                'dtype': str(dtype),
                'forward_ms': 'OOM',
                'backward_ms': 'OOM',
                'end_to_end_ms': 'OOM'
            })

        try:
            # Benchmark FlashAttention
            print("  - FlashAttention-2...")
            flash_result = benchmark_attention(
                'FlashAttention2', flash_attention_triton.apply, seq_len, d_model, dtype
            )
            results.append(flash_result)
            print(f"    Forward: {flash_result['forward_ms']:.3f}ms, Backward: {flash_result['backward_ms']:.3f}ms")
        except RuntimeError as e:
            print(f"  - FlashAttention-2 OOM: {e}")
            results.append({
                'implementation': 'FlashAttention2',
                'seq_len': seq_len,
                'd_model': d_model,
                'dtype': str(dtype),
                'forward_ms': 'OOM',
                'backward_ms': 'OOM',
                'end_to_end_ms': 'OOM'
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv('flash_attention_benchmark.csv', index=False)
    print("\n" + "="*80)
    print("Benchmark complete! Results saved to flash_attention_benchmark.csv")
    print("="*80)

    # Print summary
    print("\nSample results:")
    print(df.head(20).to_string())

    return df


if __name__ == '__main__':
    main()
