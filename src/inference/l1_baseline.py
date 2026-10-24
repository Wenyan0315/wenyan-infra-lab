"""L1 基线：跑一次"未优化"生成，记录 prefill / decode 分项耗时。

运行：python src/inference/l1_baseline.py
"""

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.tiny_gpt import TinyGPT, generate_naive

torch.manual_seed(0)

model = TinyGPT()          # 随机权重，不做任何训练
prompt_len = 20
max_new = 40
prompt = torch.randint(0, 64, (1, prompt_len))

idx, timings = generate_naive(model, prompt, max_new, temperature=1.0, top_k=8)

prefill = timings["prefill_s"]
decodes = timings["decode_s"]
n_new = idx.shape[1] - prompt_len                       # 实际新生成的 token 数
decode_total = sum(decodes)

print("== L1 未优化生成基线 ==")
print(f"prompt len        : {prompt_len}")
print(f"目标生成 token 数 : {max_new}")
print(f"实际生成 token 数 : {n_new}")
print(f"输出序列总长度    : {idx.shape[1]}")
print()
print(f"TTFT (=prefill)   : {prefill * 1000:8.2f} ms")
print(f"decode 总耗时     : {decode_total * 1000:8.2f} ms")
print(f"TPOT (decode 平均): {statistics.mean(decodes) * 1000:8.2f} ms/token")
print(f"decode 第一步     : {decodes[0] * 1000:8.2f} ms")
print(f"decode 最后一步   : {decodes[-1] * 1000:8.2f} ms")
print(f"总耗时            : {(prefill + decode_total) * 1000:8.2f} ms")
print(f"输出吞吐          : {len(decodes) / decode_total:8.1f} tokens/s")
print()
print("decode 每步耗时(ms)：")
for i, d in enumerate(decodes, start=1):
    print(f"  step {i:2d}: {d * 1000:8.3f}")