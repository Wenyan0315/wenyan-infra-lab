"""L2 数值一致性验证：带 KV Cache 的 decode 与 naive 版是否产出完全相同的结果。

运行：.venv/bin/python src/inference/l2_kvcache.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.tiny_gpt import TinyGPT
from src.inference.kv_cache import prefill, decode_step

torch.manual_seed(0)

model = TinyGPT()
prompt = torch.randint(0, 64, (1, 20))
max_new = 30

# ---- naive：每步把整个序列喂进去，取最后位置 logits，贪心 argmax ----
naive_idx = prompt.clone()
naive_logits = []
for _ in range(max_new):
    logits = model(naive_idx)[:, -1, :]
    naive_logits.append(logits)
    next_id = logits.argmax(dim=-1, keepdim=True)
    naive_idx = torch.cat([naive_idx, next_id], dim=1)

# ---- KV Cache：prefill 一次，之后每步只算新 token，贪心 argmax ----
kv_idx = prompt.clone()
kv_logits = []
last_logits, cache = prefill(model, prompt)
kv_logits.append(last_logits)
next_id = last_logits.argmax(dim=-1, keepdim=True)
kv_idx = torch.cat([kv_idx, next_id], dim=1)
for step in range(1, max_new):
    logits = decode_step(model, kv_idx[:, -1:], kv_idx.shape[1] - 1, cache)
    kv_logits.append(logits)
    next_id = logits.argmax(dim=-1, keepdim=True)
    kv_idx = torch.cat([kv_idx, next_id], dim=1)

# ---- 对比 ----
max_err = max((a - b).abs().max().item() for a, b in zip(naive_logits, kv_logits))
same_tokens = torch.equal(naive_idx, kv_idx)

print("== L2 数值一致性 ==")
print(f"步数            : {max_new}")
print(f"每步 logits 最大绝对误差 : {max_err:.3e}")
print(f"贪心 token 序列完全相同  : {same_tokens}")
print()
print("naive 序列 :", naive_idx.squeeze().tolist())
print("kv    序列 :", kv_idx.squeeze().tolist())