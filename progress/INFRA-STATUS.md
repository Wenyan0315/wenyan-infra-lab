# 推理优化线：学习状态（INFRA-STATUS）

更新时间：2026-10-22（L1 结束）

## 环境
- 目录：my_project3（推理线独立，与训练线仓库 my_project2/wenyan-llm-lab 分离）
- Python 3.14.7 / PyTorch 2.14.1 / MPS: True
- 依赖锁定：requirements.txt（torch 2.14.1、numpy 2.5.4 及传递依赖）
- 本机 Apple M4 Pro、48 GiB 统一内存，无 NVIDIA GPU

## 当前状态
- 里程碑：本次从零启动推理线
- L1 性能模型与基线：**done**（2026-10-22）
- 下一课：L2 手写 KV Cache
- 运行中的任务：无

## 关键概念掌握表

| 概念 | 状态 | 证据 | 下次变式题 |
|---|---|---|---|
| prefill / decode 两阶段 | assisted | 能说出 prefill 一次算全、decode 逐 token 生成 | 给一段 30 token prompt 生成 10 token，指出 prefill 算了哪些、decode 循环几次 |
| 生成为什么越来越慢（O(T²)） | assisted | 预测"越来越慢"方向对；理由"cache 搬运"措辞不准→实为每步重算全序列 | 手写 attention 的 q@k^T 说明 T 翻倍时计算量变几倍 |
| 测量纪律（规模 / 多次中位数 / 预热） | assisted | 变式 A 答对；但复述"三件事"只说出"T 变长"，漏预热+中位数 | 只看一组抖动数据判断该不该多次跑 |

## L1 实验记录

- `l1_baseline.py`：T=20→59，decode 每步 0.6~1.0ms 基本平坦——O(T²) 被固定开销+计时抖动淹没（真实计算只有微秒级）。
- `l1_trend.py`：block_size=512、T=20→419、预热 + 5 轮取中位数，seq_len 71→419 耗时 1.03→3.91ms，O(T²) 趋势清晰显现。
- 排错：l1_trend 首次运行报 `generate_naive() got an unexpected keyword argument 'max_new'`（形参是 `max_new_tokens`），修正后通过。

## 预测与失误

- 预测 decode 每步越来越慢：**方向对**；理由是"每次 cache 搬运越来越多"——措辞不准确，当前实现无 cache，是每步重算越来越长的历史序列。
- L1 基线前预测"趋势会显现"，实测未显现 → 本课核心差异点：规模太小 + 单次计时噪声大，把 O(T²) 的信号埋掉。
- 变式题 A（放大 T + 预热 + 中位数）**答对**。

## 仍不懂 / 待补

- "参数规模如何定量影响真实计算量"目前只有直觉，未列公式（留到 L3 prefill/decode 分项 + roofline 补）。
- TTFT / TPOT / 吞吐 / 尾延迟的正式定义与 p50/p95 测量方法还没练（L1 只用到"平均耗时"，M10 正式做）。

## 下一课

L2 手写 KV Cache：把"每步重算全序列"改成"缓存每层 K/V，每步只算新 token"，用数值一致性证明没算错，并做显存账本。