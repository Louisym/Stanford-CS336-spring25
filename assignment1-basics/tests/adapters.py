from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from cs336_basics.model.modules import Linear
    linear_layer = Linear(
        in_features=d_in,
        out_features=d_out,
        device=in_features.device,
        dtype=in_features.dtype
    )
    if weights is not None:
        weight_state = {'weight': weights.to(device=in_features.device, dtype=in_features.dtype)}
        linear_layer.load_state_dict(weight_state, strict=True)
    linear_layer.eval()
    with torch.no_grad():
        out = linear_layer(in_features)
    return out




def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    from cs336_basics.model.modules import Embedding
    embedding_layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model
    )
    if weights is not None:
        embed_state = {'embed_weight': weights.to(device=token_ids.device)}
        embedding_layer.load_state_dict(embed_state, strict=True)
    embedding_layer.eval()
    with torch.no_grad():
        out = embedding_layer(token_ids)
    return out


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    from cs336_basics.model.modules import SwiGLU
    layer = SwiGLU(d_model=d_model, d_ff=d_ff,
                   device=in_features.device, dtype=in_features.dtype)
    # 加载提供的权重
    with torch.no_grad():
        layer.w1.copy_(w1_weight.to(in_features.device, in_features.dtype))
        layer.w2.copy_(w2_weight.to(in_features.device, in_features.dtype))
        layer.w3.copy_(w3_weight.to(in_features.device, in_features.dtype))
    layer.eval()
    with torch.no_grad():
        return layer(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    from cs336_basics.model.modules import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.model.modules import multihead_self_attention
    MHA = multihead_self_attention(
        d_model=d_model, 
        num_heads=num_heads,
        device=in_features.device,
        dtype=in_features.dtype,
        use_causal_mask=True
    )
    
    with torch.no_grad():
        MHA.w_q.weight.copy_(q_proj_weight.to(in_features.device, in_features.dtype))
        MHA.w_k.weight.copy_(k_proj_weight.to(in_features.device, in_features.dtype))
        MHA.w_v.weight.copy_(v_proj_weight.to(in_features.device, in_features.dtype))
        MHA.w_o.weight.copy_(o_proj_weight.to(in_features.device, in_features.dtype))
    
    MHA.eval()
    with torch.no_grad():
        return MHA(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.model.modules import multihead_self_attention, RotaryPositionalEmbedding
    MHA = multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        position_embedding=RotaryPositionalEmbedding,
        max_seq_len=max_seq_len,
        theta=theta,
        token_positions=token_positions,
        device=in_features.device,
        dtype=in_features.dtype
    )
    
    with torch.no_grad():
        MHA.w_q.weight.copy_(q_proj_weight.to(in_features.device, in_features.dtype))
        MHA.w_k.weight.copy_(k_proj_weight.to(in_features.device, in_features.dtype))
        MHA.w_v.weight.copy_(v_proj_weight.to(in_features.device, in_features.dtype))
        MHA.w_o.weight.copy_(o_proj_weight.to(in_features.device, in_features.dtype))
    
    MHA.eval()
    with torch.no_grad():
        return MHA(in_features)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from cs336_basics.model.modules import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    from cs336_basics.model.transformer import transformer_block as TB

    block = TB(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta                # 让 Block 内部启用 RoPE
    )

    # 1) 做 key 映射
    key_map = {
        # attention
        "attn.q_proj.weight": "attn.w_q.weight",
        "attn.k_proj.weight": "attn.w_k.weight",
        "attn.v_proj.weight": "attn.w_v.weight",
        "attn.output_proj.weight": "attn.w_o.weight",
        # rmsnorm
        "ln1.weight": "ln1.g_weight",
        "ln2.weight": "ln2.g_weight",
        # ffn
        # 若你的 FFN 里是 nn.Linear，则是 ".weight"；若你自己是 nn.Parameter，就没有 ".weight"
        "ffn.w1.weight": "ffn.w1.weight" if hasattr(block.ffn, "w1") and hasattr(block.ffn.w1, "weight") else "ffn.w1",
        "ffn.w2.weight": "ffn.w2.weight" if hasattr(block.ffn, "w2") and hasattr(block.ffn.w2, "weight") else "ffn.w2",
        "ffn.w3.weight": "ffn.w3.weight" if hasattr(block.ffn, "w3") and hasattr(block.ffn.w3, "weight") else "ffn.w3",
    }

    remapped = {}
    for k_src, v in weights.items():
        k_dst = key_map.get(k_src, None)
        if k_dst is not None:
            remapped[k_dst] = v.to(device=in_features.device, dtype=in_features.dtype)

    # 2) 严格加载，防止漏加载
    block.load_state_dict(remapped, strict=False)

    block.eval()
    with torch.no_grad():
        # 3) 如果 forward 需要 token_positions，就构造并传入
        b, s, _ = in_features.shape
        token_positions = torch.arange(s, device=in_features.device).unsqueeze(0).expand(b, s)
        try:
            return block(in_features, token_positions=token_positions)
        except TypeError:
            # 如果你的 forward 不接受 token_positions，说明 Block 自己会内部处理（比如缓存或在 __init__ 里注册）
            return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    from cs336_basics.model.transformer import transformer_lm 
    import re

    # 1) Build model skeleton
    model = transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        rope_theta=rope_theta,
        d_ff=d_ff,
    )

    # 2) Build a remapped state_dict from provided reference weights
    #    Reference keys use names like:
    #    - token_embeddings.weight
    #    - layers.{i}.attn.q_proj.weight / k_proj.weight / v_proj.weight / output_proj.weight
    #    - layers.{i}.ln1.weight / ln2.weight
    #    - layers.{i}.ffn.w1.weight / w2.weight / w3.weight
    #    - ln_final.weight
    #    - lm_head.weight

    # Fetch target keys that actually exist in our implementation
    target_keys = set(model.state_dict().keys())

    remapped: dict[str, torch.Tensor] = {}

    def try_put(dst_key: str, tensor: torch.Tensor):
        # Helper: only insert if the target key exists
        if dst_key in target_keys:
            remapped[dst_key] = tensor
            return True
        return False

    # Map token embedding
    tok_src_key = None
    if "token_embeddings.weight" in weights:
        tok_src_key = "token_embeddings.weight"
    elif "tok_embeddings.weight" in weights:
        tok_src_key = "tok_embeddings.weight"

    if tok_src_key is not None:
        w = weights[tok_src_key].to(dtype=torch.float32)
        # try common destinations in our implementations
        placed = (
            try_put("token_embedding.embed_weight", w) or
            try_put("token_embeddings.embed_weight", w) or
            try_put("tok_embed.embed_weight", w) or
            try_put("embed.embed_weight", w) or
            try_put("embedding.embed_weight", w)
        )
        if not placed:
            # Fallback: find any single key that endswith 'embed_weight'
            for k in target_keys:
                if k.endswith("embed_weight"):
                    remapped[k] = w
                    placed = True
                    break

    # Per-layer mappings
    layer_pat = re.compile(r"^layers\.(\d+)\.")

    for src_key, tensor in weights.items():
        m = layer_pat.match(src_key)
        if not m:
            continue
        li = int(m.group(1))
        tail = src_key[m.end() :]
        t = tensor.to(dtype=torch.float32)

        # Attention projections
        if tail == "attn.q_proj.weight":
            # candidates in our impl
            candidates = [
                f"layers.{li}.attn.w_q.weight",
                f"blocks.{li}.attn.w_q.weight",
            ]
            for k in candidates:
                if try_put(k, t):
                    break
        elif tail == "attn.k_proj.weight":
            for k in [f"layers.{li}.attn.w_k.weight", f"blocks.{li}.attn.w_k.weight"]:
                if try_put(k, t):
                    break
        elif tail == "attn.v_proj.weight":
            for k in [f"layers.{li}.attn.w_v.weight", f"blocks.{li}.attn.w_v.weight"]:
                if try_put(k, t):
                    break
        elif tail == "attn.output_proj.weight":
            for k in [f"layers.{li}.attn.w_o.weight", f"blocks.{li}.attn.w_o.weight"]:
                if try_put(k, t):
                    break

        # RMSNorm weights
        elif tail == "ln1.weight":
            for k in [f"layers.{li}.ln1.g_weight", f"blocks.{li}.ln1.g_weight"]:
                if try_put(k, t):
                    break
        elif tail == "ln2.weight":
            for k in [f"layers.{li}.ln2.g_weight", f"blocks.{li}.ln2.g_weight"]:
                if try_put(k, t):
                    break

        # FFN (handle Linear vs Parameter-style)
        elif tail == "ffn.w1.weight":
            for k in [
                f"layers.{li}.ffn.w1.weight",
                f"blocks.{li}.ffn.w1.weight",
                f"layers.{li}.ffn.w1",
                f"blocks.{li}.ffn.w1",
            ]:
                if try_put(k, t):
                    break
        elif tail == "ffn.w2.weight":
            for k in [
                f"layers.{li}.ffn.w2.weight",
                f"blocks.{li}.ffn.w2.weight",
                f"layers.{li}.ffn.w2",
                f"blocks.{li}.ffn.w2",
            ]:
                if try_put(k, t):
                    break
        elif tail == "ffn.w3.weight":
            for k in [
                f"layers.{li}.ffn.w3.weight",
                f"blocks.{li}.ffn.w3.weight",
                f"layers.{li}.ffn.w3",
                f"blocks.{li}.ffn.w3",
            ]:
                if try_put(k, t):
                    break

    # Final norm
    if "ln_final.weight" in weights:
        t = weights["ln_final.weight"].to(dtype=torch.float32)
        for k in ["output_norm.g_weight", "ln_final.g_weight", "norm_final.g_weight", "final_norm.g_weight"]:
            if try_put(k, t):
                break

    # LM head
    lm_src_key = None
    if "lm_head.weight" in weights:
        lm_src_key = "lm_head.weight"
    elif "output.weight" in weights:
        lm_src_key = "output.weight"
    elif "unembed.weight" in weights:
        lm_src_key = "unembed.weight"

    if lm_src_key is not None:
        t = weights[lm_src_key].to(dtype=torch.float32)
        for k in ["output_embedding.weight", "lm_head.weight", "output_head.weight", "head.weight"]:
            if try_put(k, t):
                break

    # 3) Load with strict=False (some buffers like RoPE may be non-persistent)
    model.load_state_dict(remapped, strict=False)

    # Optionally tie output head to token embedding if head wasn't provided
    try:
        # only tie if destination head wasn't in remapped and both parameters exist
        need_tie = True
        for cand in ["output_embedding.weight", "lm_head.weight", "output_head.weight", "head.weight"]:
            if cand in remapped and cand in target_keys:
                need_tie = False
                break
        if need_tie:
            # try common attribute paths
            tok_param = None
            head_param = None
            # fetch actual module attributes if present
            if hasattr(model, "token_embedding") and hasattr(model.token_embedding, "weight"):
                tok_param = model.token_embedding.weight
            elif hasattr(model, "token_embeddings") and hasattr(model.token_embeddings, "embed_weight"):
                tok_param = model.token_embeddings.embed_weight
            elif hasattr(model, "embedding") and hasattr(model.embedding, "embed_weight"):
                tok_param = model.embedding.embed_weight

            if hasattr(model, "output_embedding") and hasattr(model.output_embedding, "weight"):
                head_param = model.output_embedding.weight
            elif hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                head_param = model.lm_head.weight

            if tok_param is not None and head_param is not None and tok_param.shape == head_param.shape:
                # share storage (no copy)
                with torch.no_grad():
                    head_param.set_(tok_param)
    except Exception:
        pass

    # 4) Forward
    model.eval()
    with torch.no_grad():
        return model(in_indices)
    

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from cs336_basics.model.modules import RMSNorm
    rmsnorm_layer = RMSNorm(
        d_model=d_model,
        eps=eps
    )
    if weights is not None:
        g_state = {'g_weight': weights.to(device=in_features.device)}
        rmsnorm_layer.load_state_dict(g_state, strict=True)
    return rmsnorm_layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    from cs336_basics.trainer.data_loading import data_loading
    return data_loading(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from cs336_basics.model.modules import softmax
    return softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    from cs336_basics.trainer.utils import cross_entropy
    #note that this 'batch_size' is actually batch_size*seq_len.
    #in order to effciently calculate the crossentropy(where there is a batch_size*seq_len in the denomitor)
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    from cs336_basics.trainer.utils import gradient_clipping
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    from cs336_basics.trainer.AdamW import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    from cs336_basics.trainer.utils import learning_rate_schedule
    return learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    from cs336_basics.check_pointing import save_checkpoint
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    from cs336_basics.check_pointing import load_checkpoint
    return load_checkpoint(src, model, optimizer)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cs336_basics.tokenizer import Tokenizer
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)

from collections import Counter
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    #initialize the vocabulary
    vocab = _init_vocab({}, special_tokens)
    #pretokenization
    cnt_pretokens = Counter()
    with open(input_path, 'r', encoding='UTF-8') as f:
        text = f.read()
    chunked_text = pre_tokenization(text, special_tokens)
    for word in chunked_text:
        cnt_pretokens[word_2_byte(word)] += 1
    #merge
    merge_rule = []
    while len(vocab) < vocab_size:
        pair_cnt = Counter()
        for pretoken, cnt in cnt_pretokens.items():
            #pretoken e.g. (b'a', b'b', b'x0f8')
            for i in range(len(pretoken)-1):
                pair = (pretoken[i], pretoken[i+1])
                pair_cnt[pair] += cnt
        if not pair_cnt:
            break
        max_cnt = max(pair_cnt.values())
        candidate = [p for p,cnt in pair_cnt.items() if cnt == max_cnt]
        merge_pair = max(pair_cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merge_rule.append(merge_pair)
        n = len(vocab)
        new_token = merge_pair[0] + merge_pair[1]
        vocab[n] = new_token
        #now we can apply the merge to tokens
        change = []
        for pretoken, cnt in cnt_pretokens.items():
            start_idx = [i for i in range(len(pretoken)-1) if pretoken[i:i+2] == merge_pair]
            if start_idx:
                i = 0
                new_pre_token = []
                while i < len(pretoken):
                    if i in start_idx:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(pretoken[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                change.append([new_pre_token, pretoken, cnt])
        if not change:
            break
        for new_t, old_t, cnt in change:
            cnt_pretokens[new_t] += cnt
            cnt_pretokens[old_t] -= cnt
            if cnt_pretokens[old_t] <= 0:
                del cnt_pretokens[old_t]

    return vocab, merge_rule

                    


    
    
    

import regex as re
def pre_tokenization(s: str, special_token: list[str]) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # ① 没有 special 也要按正则切
    if not special_token:
        return re.findall(PAT, s)

    # ③ 长→短排序，防止短的抢先匹配
    toks = sorted(special_token, key=len, reverse=True)
    union = "|".join(re.escape(t) for t in toks)
    parts = re.split(f"({union})", s)

    out = []
    st = set(special_token)
    for part in parts:
        if not part:
            continue
        # ② special 只作为边界，完全跳过
        if part in st:
            continue
        out.extend(re.findall(PAT, part))
    return out

# #multiprocessing's pretoken worker
# def pretoken_worker(input_path, start, end, special_tokens, out_q):
#     import regex as re
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         data = f.read(end - start)
#     text = data.decode("utf-8")  # 严格解码

#     # special 处理（长度降序 + re.escape）
#     toks = sorted(special_tokens, key=len, reverse=True)
#     union = "|".join(re.escape(t) for t in toks)
#     parts = re.split(f"({union})", text)

#     from collections import Counter
#     cnt = Counter()
#     for part in parts:
#         if not part or part in special_tokens:
#             continue
#         for m in re.finditer(PAT, part):
#             token = tuple(word_2_byte(m.group(0)))
#             cnt[token] += 1

#     out_q.put(cnt)

def _init_vocab(vocab: dict, special_token:list):
    special_token_encoded = [s.encode('UTF-8') for s in special_token]
    idx = 0
    for code in special_token_encoded:
        vocab[idx] = code
        idx += 1
    
    for i in range(256):
        init_str = bytes([i])
        if init_str not in vocab.values():
            vocab[idx] = init_str
            idx += 1
    return vocab

def word_2_byte(word: str) -> tuple[bytes, ...]:
    word_decoded = list(word.encode('UTF-8'))
    #split the bytes
    word_byte = [bytes([b]) for b in word_decoded]
    return tuple(word_byte)