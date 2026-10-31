"""KV Cache：复用 TinyGPT 权重，decode 每步只算新 token，历史 K/V 从缓存读取。

与 naive 版数学等价——"数值一致性"是 L2 的验收点：
同一模型、同一 prompt、贪心采样下，带缓存与不带缓存应产出完全相同的 token 序列。
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


def _qkv(out: torch.Tensor, B: int, T: int, n_head: int):
    """把 c_attn 的输出 [B, T, 3C] 拆成 q/k/v，各自 reshape 成 [B, n_head, T, head_dim]。"""
    C = out.shape[-1] // 3              # C = n_embd
    q, k, v = out.split(C, dim=-1)
    head_dim = C // n_head
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)
    return q, k, v


@torch.no_grad()
def prefill(model, idx: torch.Tensor):
    """一次性处理 prompt 全部 token，缓存各层 K/V，返回 (最后位置 logits, cache)。

    cache[i] = (k, v)，形状各为 [B, n_head, T, head_dim]。
    """
    B, T = idx.shape
    device = idx.device

    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=device))
    cache = []
    for block in model.blocks:
        xh = block.ln1(x)
        q, k, v = _qkv(block.attn.c_attn(xh), B, T, block.attn.n_head)
        cache.append((k, v))

        att = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)  # [B, nh, T, T]
        att = att.masked_fill(block.attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v                                                     # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))

    logits = model.lm_head(model.ln_f(x))                                # [B, T, V]
    return logits[:, -1, :], cache


@torch.no_grad()
def decode_step(model, token: torch.Tensor, pos: int, cache):
    """处理 1 个新 token（绝对位置 pos），更新 cache，返回该位置的 logits。

    关键：只算新 token 的 q/k/v，新 q 去查"缓存里的历史 k/v + 刚 append 的新 k/v"。
    """
    B = token.shape[0]
    device = token.device

    x = model.tok_emb(token) + model.pos_emb(torch.tensor([pos], device=device))
    for i, block in enumerate(model.blocks):
        xh = block.ln1(x)
        q, k, v = _qkv(block.attn.c_attn(xh), B, 1, block.attn.n_head)

        k_old, v_old = cache[i]
        k = torch.cat([k_old, k], dim=2)              # [B, nh, T+1, hd]
        v = torch.cat([v_old, v], dim=2)
        cache[i] = (k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)  # [B, nh, 1, T+1]
        att = F.softmax(att, dim=-1)                 # 无需 mask：新 token 天然只能看到 ≤ 自己的历史

        y = att @ v                                                       # [B, nh, 1, hd]
        y = y.transpose(1, 2).contiguous().view(B, 1, -1)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))

    logits = model.lm_head(model.ln_f(x))              # [B, 1, V]
    return logits[:, -1, :]


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None):
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)     # [1, 1]


@torch.no_grad()
def generate_kv(model, idx: torch.Tensor, max_new_tokens: int,
                temperature: float = 1.0, top_k: int | None = None):
    """带 KV Cache 的生成，接口与 generate_naive 对齐，返回 (idx, timings)。"""
    model.eval()

    t0 = time.perf_counter()
    last_logits, cache = prefill(model, idx)
    next_id = _sample(last_logits, temperature, top_k)
    t1 = time.perf_counter()
    prefill_s = t1 - t0

    idx = torch.cat([idx, next_id], dim=1)
    decode_s: list[float] = []
    for _ in range(1, max_new_tokens):
        t0 = time.perf_counter()
        logits = decode_step(model, idx[:, -1:], idx.shape[1] - 1, cache)
        next_id = _sample(logits, temperature, top_k)
        t1 = time.perf_counter()
        decode_s.append(t1 - t0)

        idx = torch.cat([idx, next_id], dim=1)
        if idx.shape[1] >= model.block_size:
            break

    return idx, {"prefill_s": prefill_s, "decode_s": decode_s}