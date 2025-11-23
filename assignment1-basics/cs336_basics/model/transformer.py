
import torch
import torch.nn as nn
import cs336_basics.model.modules as modules
from  cs336_basics.model.modules import SwiGLU as FFN

class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        """
        d_model: int  Dimensionality of the Transformer block inputs.
        num_heads: int  Number of heads to use in multi-head self-attention.
        d_ff: int  Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Names chosen to match the snapshot/state_dict expected by tests
        self.ln1 = modules.RMSNorm(d_model)
        self.ln2 = modules.RMSNorm(d_model)

        # Attention submodule named "attn" to match expected keys like "attn.q_proj.weight" (after mapping)
        self.attn = modules.multihead_self_attention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        # Make RoPE buffer non-persistent so tests don't require providing it in state_dict
        # (prevents missing key like "attn.pe.inv_freq" with strict=True)
        if hasattr(self.attn, "pe") and hasattr(self.attn.pe, "inv_freq"):
            buf = self.attn.pe.inv_freq
            # re-register buffer as non-persistent if it exists
            try:
                # remove then re-register as non-persistent
                del self.attn.pe._buffers["inv_freq"]
                self.attn.pe.register_buffer("inv_freq", buf, persistent=False)
            except Exception:
                # if anything goes wrong, just keep original; loading with strict mapping may still work
                pass

        self.ffn = FFN(d_model=d_model, d_ff=d_ff)

        # Optional: move to device/dtype if provided
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Input tensor shape: (batch, seq_len, d_model)
        Pre-norm Transformer block:
            x = x + Attn( LN1(x) )
            x = x + FFN( LN2(x) )
        """
        x = in_features
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    

class transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, rope_theta: float, d_ff: int):
        '''
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
        embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of
        the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding = modules.Embedding(vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            [
                transformer_block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = modules.RMSNorm(d_model)
        self.output_embedding = modules.Linear(in_features=d_model, out_features=vocab_size)
    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)            # (batch, seq, d_model)
        for layer in self.layers:              # stack of Transformer blocks
            x = layer(x)
        x = self.output_norm(x)                # pre-logits norm
        x = self.output_embedding(x)           # project to vocab logits
        return x                               # return logits (no softmax)
