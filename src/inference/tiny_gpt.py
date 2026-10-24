"""最小随机权重 GPT：仅供推理线做性能/数值测试底座（不训练、不展开讲结构）。

这是"未优化"实现——生成时每步都把整个序列重新 forward 一遍（无 KV Cache），
L2 会拿它做优化对比。随机权重下输出是乱码，这里只关心计时与数值正确性。
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。qkv 用一次合并投影，方便后续拆出 K/V 做缓存。"""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # 因果遮罩：位置 i 只能看 j <= i（下三角为 1）
        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("bias", tril.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)                 # 各 [B, T, C]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 手写 QK^T / sqrt(d) + softmax，不调 SDPA，便于 L2 与缓存版逐位对照
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v                                               # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        n_vocab: int = 64,
        n_embd: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        block_size: int = 64,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)                                   # [B, T, V]


@torch.no_grad()
def generate_naive(model: nn.Module, idx: torch.Tensor, max_new_tokens: int,
                   temperature: float = 1.0, top_k: int | None = None):
    """未优化生成：每步把整个序列重算一遍。

    返回 (完整 token 序列, {"prefill_s": float, "decode_s": list[float]})。
    prefill_s 是第一次 forward 的耗时（含采样出第一个 token）；
    decode_s 里的每一项是"生成剩余 token 时"每步的耗时。
    注意：这里是 naive 实现，decode 每一步都要重算整段序列，所以越到后面越慢。
    """
    model.eval()
    prefill_s = None
    decode_s: list[float] = []

    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        logits = model(idx)[:, -1, :] / temperature                # [1, V]
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)          # [1, 1]
        t1 = time.perf_counter()

        if step == 0:
            prefill_s = t1 - t0
        else:
            decode_s.append(t1 - t0)

        idx = torch.cat([idx, next_id], dim=1)
        if idx.shape[1] >= model.block_size:
            break

    return idx, {"prefill_s": prefill_s, "decode_s": decode_s}