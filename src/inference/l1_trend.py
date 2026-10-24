"""L1 趋势验证：加大序列长度 + 预热 + 多轮取中位数，让 O(T²) 显现出来。

运行：.venv/bin/python src/inference/l1_trend.py
"""

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.tiny_gpt import TinyGPT, generate_naive

torch.manual_seed(0)

# block_size 从默认 64 放大，否则生成到 60 就会被截断
model = TinyGPT(block_size=512)
prompt_len = 20
max_new = 400
prompt = torch.randint(0, 64, (1, prompt_len))

# 预热：先跑一小段，排除首次 forward 的冷启动
_, _ = generate_naive(model, prompt, max_new_tokens=50, top_k=8)

rounds = 5
all_decodes = []
for _ in range(rounds):
    _, t = generate_naive(model, prompt, max_new, top_k=8)
    all_decodes.append(t["decode_s"])

n = len(all_decodes[0])
# 对"同一个 step 位置"跨 5 轮取中位数，抹掉单次抖动
medians = [
    statistics.median(all_decodes[r][i] for r in range(rounds))
    for i in range(n)
]

print("== 放大 T + 预热 + 5 轮中位数后的 decode 每步耗时 ==")
print(f"序列长度范围: {prompt_len} -> {prompt_len + n}")
print()
print("观察点 (seq_len, 5轮中位数耗时 ms):")
for i in [0, 25, 50, 100, 200, 300, 350, n - 1]:
    print(f"  seq_len≈{prompt_len + 1 + i:4d}    {medians[i] * 1000:9.3f} ms")