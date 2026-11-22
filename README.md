# Wenyan Inference Lab

用一个小型 GPT 学习语言模型推理。先测朴素解码的耗时，再实现 KV Cache，用逐步 logits 和贪心生成结果核对缓存版的数值一致性。

| 实验 | 代码 | 观察内容 |
|---|---|---|
| L1 · 性能基线 | [基线脚本](src/inference/l1_baseline.py) · [长度趋势](src/inference/l1_trend.py) | 序列变长时，每步解码耗时如何变化 |
| L2 · 手写 KV Cache | [缓存实现](src/inference/kv_cache.py) · [一致性验证](src/inference/l2_kvcache.py) | 缓存版与朴素版的 logits、生成 token 和 KV 缓存容量 |

## 运行实验

仓库的[依赖锁定文件](requirements.txt)对应原始实验环境（macOS、Python 3.14.7）。从仓库根目录执行。

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/inference/l1_baseline.py
python src/inference/l2_kvcache.py
```

[实验进度与实测值](progress/INFRA-STATUS.md)记录了 L1、L2 的结果和下一步的 prefill / decode 分项测量。模型训练与 tokenizer 实验见 [Wenyan LLM Lab](https://github.com/Wenyan0315/wenyan-llm-lab)。
