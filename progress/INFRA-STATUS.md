# 推理优化线：学习状态（INFRA-STATUS）

更新时间：2026-10-31（L2 结束）

## 环境
- 目录：my_project3（推理线独立，与训练线仓库 my_project2/wenyan-llm-lab 分离）
- Python 3.14.7 / PyTorch 2.14.1 / MPS: True
- 依赖锁定：requirements.txt（torch 2.14.1、numpy 2.5.4 及传递依赖）
- 本机 Apple M4 Pro、48 GiB 统一内存，无 NVIDIA GPU
- GitHub：github.com/Wenyan0315/wenyan-infra-lab（public，每课结束 push）

## 当前状态
- L1 性能模型与基线：**done**（2026-10-22）
- L2 手写 KV Cache：**done**（2026-10-31）
- 下一课：L3 prefill / decode 分项测量
- 运行中的任务：无

## 关键概念掌握表

| 概念 | 状态 | 证据 | 下次变式题 |
|---|---|---|---|
| prefill / decode 两阶段 | assisted | 能说出 prefill 一次算全、decode 逐 token 生成 | 给一段 30 token prompt 生成 10 token，指出 prefill 算了哪些、decode 循环几次 |
| O(T²) 与测量纪律 | assisted | 变式 A 答对；复述"三件事"只说出 T 变长，漏预热+中位数 | 只看一组抖动数据判断该不该多次跑 |
| KV Cache 数值一致性 | assisted | 预测 A（输出一致）对；实测 logits 最大误差 9.5e-7、token 完全一致 | 解释"cache 只省重复计算、不改结果" |
| KV 显存账本 | assisted | 玩具模型满上下文算对 1 MiB；大模型例子数值差 10 倍，锚点法纠正 | 只给模型参数，报 KV cache 字节数 |

## L2 实验记录

- `kv_cache.py`：新增 `prefill` / `decode_step` / `generate_kv`，复用 TinyGPT 权重，decode 每步只算新 token，历史 K/V 从缓存 append。
- `l2_kvcache.py`：贪心 argmax 下 naive vs KV 对照，30 步每步 logits 最大绝对误差 9.537e-07，token 序列完全相同。
- 显存账本公式 `KV bytes = 2 × layers × B × T × heads × head_dim × dtype`：玩具模型满上下文（B=1, T=512, fp32）= 1 MiB；近真实配置（28 层 × 16 头 × 128 维，B=8, T=4096, bf16）≈ 7.5 GB。

## 预测与失误

- L2 数值一致性预测 A（输出一致）**对**，理由"cache 只优化计算、不改数学结果"准确。
- 显存账本 1MB **算对**；但对"为什么小模型感觉不到省时间"的解释混了两个维度——把"缓存占得少（显存）"误当"省时感觉不到（计算）"的原因，经校准：省时收益取决于真实计算量大小，模型小则微秒级被固定开销淹没。
- 大模型 KV cache 例子算成 75 GB，实际 7.5 GB（差 10 倍），用"每 token ≈224 KB"锚点纠正。

## 仍不懂 / 待补

- "参数规模如何定量影响真实计算量"仍只有直觉，未列公式（L3 roofline 补）。
- TTFT / TPOT / 吞吐 / 尾延迟 的正式定义与 p50/p95 测量方法（M10 正式做）。
- KV cache 实际后端额外开销（padding、内存对齐、碎片等）留到 llama.cpp / vLLM 实测补。

## 下一课

L3 prefill / decode 分项测量：为什么 decode 慢（memory-bound vs compute-bound），roofline 直觉，用实测区分两阶段的性能特征。