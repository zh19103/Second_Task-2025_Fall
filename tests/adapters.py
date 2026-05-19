from __future__ import annotations

import sys
import os
import importlib
import math
import numpy as np
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.typing as npt
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

current_working_dir = os.getcwd()
if current_working_dir not in sys.path:
    sys.path.append(current_working_dir)

try:
    import submission
    importlib.reload(submission)
    HAS_SUBMISSION = True
except ImportError:
    HAS_SUBMISSION = False
    print(f"Warning: submission.py not found in {current_working_dir}, using default/empty implementation.")


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    return torch.matmul(in_features, weights.t())


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    # 索引查找实现
    return torch.index_select(weights, 0, token_ids.reshape(-1)).reshape(*token_ids.shape, d_model)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, "d_ff d_model"],
    w2_weight: Float[Tensor, "d_model d_ff"],
    w3_weight: Float[Tensor, "d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    x_w1 = run_linear(d_model, d_ff, w1_weight, in_features)
    x_w3 = run_linear(d_model, d_ff, w3_weight, in_features)
    swiglu_act = run_silu(x_w1) * x_w3
    output = run_linear(d_ff, d_model, w2_weight, swiglu_act)
    return output


def run_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """
    - 统一 float32 计算
    - 被 mask 的位置直接置为 -inf
    - softmax 前 clamp 到 [-60, 60] 防止溢出
    """
    f32 = torch.float32
    out_dtype = V.dtype

    Q32 = Q.to(f32)
    K32 = K.to(f32)
    V32 = V.to(f32)

    d_k = float(Q.size(-1))
    logits = torch.matmul(Q32, K32.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        visible = mask if mask.dtype == torch.bool else (mask != 0)
        logits = torch.where(
            visible,
            logits,
            torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
        )

    probs = F.softmax(logits, dim=-1)

    out = torch.matmul(probs, V32)
    return out.to(out_dtype)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    x = in_query_or_key  # (..., heads, seq, d_k)
    out_dtype = x.dtype
    x32 = x.to(torch.float32)
    d = x32.size(-1)
    assert d % 2 == 0, "d_k must be even for RoPE"

    base = torch.tensor(float(theta), dtype=torch.float32, device=x.device)
    # inv_freq: base^{-2i/d}, i=0..d/2-1
    idx = torch.arange(0, d, 2, dtype=torch.float32, device=x.device)
    inv_freq = base.pow(-idx / float(d))

    # 标准化 positions 到 (..., seq)
    pos = token_positions.to(torch.float32)
    # 若为 (..., heads, seq)，取任一 head（各 head 相同）
    if pos.dim() == x32.dim() - 1 and pos.size(-2) == x32.size(-2):
        pos = pos[..., 0, :]
    if pos.dim() < x32.dim() - 1:
        pos = pos.unsqueeze(-2)  # (..., 1, seq)

    angles = pos.unsqueeze(-1) * inv_freq  # (..., seq, d/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 复数旋转
    even = x32[..., ::2]
    odd = x32[..., 1::2]
    xr = torch.stack([even, odd], dim=-1)                  # (..., seq, d/2, 2)
    x_complex = torch.view_as_complex(xr)                  # (..., seq, d/2)
    rot = torch.view_as_complex(torch.stack([cos, sin], dim=-1))  # (..., seq, d/2)
    x_rot = x_complex * rot
    xr_out = torch.view_as_real(x_rot)
    out = torch.empty_like(x32)
    out[..., ::2] = xr_out[..., 0]
    out[..., 1::2] = xr_out[..., 1]
    return out.to(out_dtype)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_model d_model"],
    k_proj_weight: Float[Tensor, "d_model d_model"],
    v_proj_weight: Float[Tensor, "d_model d_model"],
    o_proj_weight: Float[Tensor, "d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    mask: Bool[Tensor, " ... sequence_length sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    seq_len = in_features.size(-2)
    batch_dims = in_features.shape[:-2]
    if token_positions is None:
        base = torch.arange(seq_len, device=in_features.device, dtype=torch.long)
        token_positions = base.expand(*batch_dims, seq_len)

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_head = d_model // num_heads

    Q = run_linear(d_model, d_model, q_proj_weight.to(in_features.dtype), in_features)
    K = run_linear(d_model, d_model, k_proj_weight.to(in_features.dtype), in_features)
    V = run_linear(d_model, d_model, v_proj_weight.to(in_features.dtype), in_features)

    Q = Q.reshape(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)
    K = K.reshape(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)
    V = V.reshape(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)

    pos = token_positions.unsqueeze(-2).expand(*batch_dims, num_heads, seq_len)
    Q_rope = run_rope(d_head, theta, max_seq_len, Q, pos)
    K_rope = run_rope(d_head, theta, max_seq_len, K, pos)

    mask_expanded = None
    if mask is not None:
        if mask.dim() == len(batch_dims) + 2:
            mask_expanded = mask.unsqueeze(-3)
        elif mask.dim() == len(batch_dims) + 3:
            mask_expanded = mask
        else:
            raise ValueError(f"mask维度错误: 期望{len(batch_dims)+2}/{len(batch_dims)+3}，实际{mask.dim()}")
        mask_expanded = mask_expanded.expand(*batch_dims, num_heads, seq_len, seq_len).bool()

    attn_output = run_scaled_dot_product_attention(Q_rope, K_rope, V, mask_expanded)
    attn_output = attn_output.transpose(-3, -2).reshape(*batch_dims, seq_len, d_model)
    output = run_linear(d_model, d_model, o_proj_weight.to(attn_output.dtype), attn_output)
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_model d_model"],
    k_proj_weight: Float[Tensor, "d_model d_model"],
    v_proj_weight: Float[Tensor, "d_model d_model"],
    o_proj_weight: Float[Tensor, "d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    mask: Bool[Tensor, " ... sequence_length sequence_length"] | None = None,
    max_seq_len: int | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    if max_seq_len is None:
        max_seq_len = in_features.size(-2)
    return run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=10000.0,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_features,
        token_positions=token_positions,
        mask=mask,
    )


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
    eps: float = 1e-5,
    mask: Bool[Tensor, " batch sequence_length sequence_length"] | None = None,
    debug: bool = False,
) -> Float[Tensor, " batch sequence_length d_model"]:
    x = in_features
    b, s, _ = x.shape

    x_norm = run_rmsnorm(d_model, eps=eps, weights=weights["ln1.weight"], in_features=x)

    if mask is None:
        causal = torch.tril(torch.ones((s, s), dtype=torch.bool, device=x.device))
        mask = causal.unsqueeze(0).expand(b, s, s)

    # 显式 positions（ai说可选但更清晰）
    positions = torch.arange(s, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, s)

    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_norm,
        token_positions=positions,
        mask=mask,
    )
    x = x + attn_out

    x_norm = run_rmsnorm(d_model, eps=eps, weights=weights["ln2.weight"], in_features=x)
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=x_norm
    )
    x = x + ffn_out
    return x


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
    eps: float = 1e-5,
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    token_embeds = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices
    )
    x = token_embeds
    for layer_idx in range(num_layers):
        layer_weights = {
            "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
            "attn.q_proj.weight": weights[f"layers.{layer_idx}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{layer_idx}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{layer_idx}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{layer_idx}.attn.output_proj.weight"],
            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
            "ffn.w1.weight": weights[f"layers.{layer_idx}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{layer_idx}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{layer_idx}.ffn.w3.weight"],
        }
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
            eps=eps
        )
    x_norm = run_rmsnorm(d_model, eps=eps, weights=weights["ln_final.weight"], in_features=x)
    # 对齐 lm_head 权重 dtype
    lm_logits = run_linear(d_model, vocab_size, weights["lm_head.weight"].to(x_norm.dtype), x_norm)
    return lm_logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: torch.Tensor,           # [d_model]
    in_features: torch.Tensor,       # [..., d_model]
) -> torch.Tensor:
    # 数值稳定：先 cast 到 float32 再还原。。。或许其实根本不需要
    x = in_features
    out_dtype = x.dtype
    x32 = x.to(torch.float32)
    w32 = weights.to(torch.float32)
    msq = (x32 * x32).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(msq + eps)
    y32 = (x32 * inv_rms) * w32
    return y32.to(out_dtype)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # softmax（减最大值）
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    exp_vals = torch.exp(in_features - max_vals)
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    return exp_vals / (sum_exp + 1e-9)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    log_softmax = torch.log(run_softmax(inputs, dim=-1))
    nll_loss = -log_softmax[torch.arange(len(targets)), targets]
    avg_loss = torch.mean(nll_loss)
    return avg_loss


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    if max_l2_norm <= 0.0:
        return
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.data.mul_(clip_coef)


def get_adamw_cls() -> Any:
    return torch.optim.AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * (it + 1) / warmup_iters
    if it > warmup_iters + cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / cosine_cycle_iters
    lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    return lr


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    vocab_str = {i: b.decode("utf-8", errors="replace") for i, b in vocab.items()}
    merges_str = [(a.decode("utf-8"), b.decode("utf-8")) for a, b in merges]
    tokenizer = Tokenizer(BPE(vocab=vocab_str, merges=merges_str))
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    tokenizer.pre_tokenizer = ByteLevel()
    return tokenizer


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=kwargs.get("min_frequency", 2),
        show_progress=kwargs.get("show_progress", True)
    )
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(input_path)], trainer=trainer)
    vocab = {int(id): token.encode("utf-8") for id, token in tokenizer.get_vocab().items()}
    merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in tokenizer.model.merges]
    return vocab, merges


def run_abstopk(
    in_features: Float[Tensor, " ..."],
    k: int,
) -> Float[Tensor, " ..."]:
    abs_vals = torch.abs(in_features)
    topk_vals, topk_indices = torch.topk(abs_vals, k=k, dim=-1, sorted=False)
    mask = torch.zeros_like(in_features, dtype=torch.bool)
    mask.scatter_(-1, topk_indices, True)
    output = torch.where(mask, in_features, torch.tensor(0.0, device=in_features.device, dtype=in_features.dtype))
    return output


def run_attention_with_sink(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    sink_token: Float[Tensor, " 1 d_k"] | None = None,
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    if sink_token is None:
        return run_scaled_dot_product_attention(Q, K, V, mask)

    batch_dims = K.shape[:-2]
    device = K.device
    dtype = K.dtype

    sink_k = sink_token.to(device=device, dtype=dtype).expand(*batch_dims, 1, -1)
    K_with_sink = torch.cat([sink_k, K], dim=-2)

    dv = V.size(-1)
    dk = K.size(-1)
    sink_v = sink_k[..., :min(dk, dv)]
    if dv > sink_v.size(-1):
        pad = (0, dv - sink_v.size(-1))
        sink_v = F.pad(sink_v, pad, mode="constant", value=0.0)
    V_with_sink = torch.cat([sink_v, V], dim=-2)

    if mask is not None:
        visible = (mask != 0) if mask.dtype != torch.bool else mask
        sink_visible = torch.ones(*visible.shape[:-1], 1, dtype=visible.dtype, device=visible.device)
        mask_with_sink = torch.cat([sink_visible, visible], dim=-1)
    else:
        mask_with_sink = None

    return run_scaled_dot_product_attention(Q, K_with_sink, V_with_sink, mask_with_sink)


def run_magnitude_pruning(
    model: torch.nn.Module,
    sparsity_level: float,
) -> None:
    
    #对nn.Linear 层做幅度剪枝：
    #每个 Linear 层内独立计算阈值（按该层权重绝对值分布的 sparsity_level 分位数）
    #仅剪了 weight
    #注意：剪枝后权重被置零，但仍保留原参数结构（未真正删除连接），以保持接口一致。
    
    assert 0.0 <= sparsity_level <= 1.0, "sparsity_level must be between 0 and 1"
    if sparsity_level == 0.0:
        return

    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight
            if W is None or W.data.numel() == 0:
                continue
            with torch.no_grad():
                abs_w = W.data.abs()
                thresh = torch.quantile(abs_w.reshape(-1), q=sparsity_level)
                mask = (abs_w >= thresh).to(W.data.dtype)
                W.data.mul_(mask)
