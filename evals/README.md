# Firmware RAG Retrieval Eval

## 数据集

`data/eval/retrieval_v1.json` 共 50 题：

- 15 题 FreeRTOS Task Management。
- 15 题 ESP-IDF NVS。
- 15 题 ESP-IDF Wi-Fi。
- 5 题不在当前语料范围内，用于评测拒答。

可回答题同时标注来源和证据词组。只有 Top-K 段落来自正确来源，并且
包含每组至少一个证据词时才算命中。语料外问题只有返回
`no_evidence` 才算通过。

## 指标

- `Hit@K`：只在可回答题上统计证据是否进入 Top-K。
- `MRR`：只在可回答题上统计第一条正确证据的倒数排名。
- `abstention_accuracy`：只在语料外问题上统计正确拒答比例。
- `overall_accuracy`：可回答命中和语料外拒答共同计算。

评测 CLI 会在报告的 `run_config` 中记录检索器、Reranker、Embedding
模型、切分参数、相似度门槛，以及题集和语料目录的 SHA256。使用
`--strict-label-audit` 时，每道可回答题必须在正确来源的某一个 Chunk
中同时包含全部证据词组，否则评测会在检索前终止。

## v1 结果

固定参数：

- `top_k=3`
- `chunk_size=1200`
- `chunk_overlap=150`
- Embedding：`qwen3-embedding:0.6b`
- Vector threshold：`0.55`

| 检索器 | Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4889 | 0.4370 | 0.8000 | 0.5200 |
| Vector | 0.9556 | 0.8407 | 1.0000 | 0.9600 |
| Hybrid RRF | 0.9556 | 0.8148 | 1.0000 | 0.9600 |
| Vector + query normalization | 1.0000 | 0.8593 | 1.0000 | 1.0000 |

## 结论

1. 中文问题检索英文官方文档时，BM25 存在明显词汇鸿沟。
2. 本地向量检索显著提高召回；当前 RRF 没有进一步提升 Hit@3，MRR
   反而低于纯向量。
3. 默认 `0.25` 阈值会让所有语料外问题都返回相似段落。v1 分数分布中，
   可回答题最高分最低为 `0.619`，语料外题最高分最高为 `0.451`，因此
   暂时将门槛设为 `0.55`。
4. `0.55` 使用同一 v1 数据集探索得到，不是独立测试结论；建立留出集
   前不能把 96% 总体准确率表述为泛化准确率。
5. 两条失败题增加领域查询规范化后，同一数据集回归为满分。这只能证明
   已知失败被修复，不能表述为未知问题准确率 100%。

## 保留的失败样例

- `freertos-task-name-purpose`：正确答案位于相邻 chunk，Top-3 只命中
  API 原型和参数开头，没有命中包含 `debugging aid` 的下一段。
- `nvs-power-loss-guarantee`：Top-3 偏向 NVS 写入日志和初始化内容，
  没有命中 Security/Robustness 中的断电保证。

这两题的正确证据原本位于向量候选第 5 和第 8。项目增加了两条可解释的
领域规范化规则，并把优化结果另存为
`retrieval_v1_vector_query_expansion.json`。没有删除失败题或放宽证据
标签来制造满分；优化前报告仍保留用于对比。

## 独立留出集

查询规范化和 `0.55` 门槛固定后，新增 `data/eval/holdout_v1.json`：

- 15 条可回答题。
- 3 条语料外问题。
- Hit@3：0.8667。
- MRR：0.8000。
- 拒答准确率：1.0000。
- 总体准确率：0.8889。

首次运行原样保存在 `holdout_v1_vector_raw.json`。标注审计发现两处人工
证据标签问题：一处短语被 Markdown 换行拆开，另一处把文档中的
`Synchronization events` 错标为 `external`。修正标签后重算得到上述
最终指标，检索参数和查询规则没有调整。

最终仍保留两条真实失败：

- `holdout-wifi-all-channel-scan`
- `holdout-wifi-background-home-channel`

不再针对这两条调整当前模型，否则留出集会被污染成新的开发集。

## 私有业务语料留出集

私有语料的检索器、切分参数和 `0.55` 门槛冻结后，新增 20 条自然问法：

- 15 条可回答题，覆盖连接、任务、功率、协议、升级和配置同步。
- 5 条业务相邻但语料未覆盖的困难拒答题。
- 每道可回答题都通过同 Chunk 严格证据标签审计。

首次运行原样保存在 Git 忽略目录中的
`var/private-corpus/holdout-v1-vector-raw.json`：

| Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| ---: | ---: | ---: | ---: |
| 0.9333 | 0.7444 | 1.0000 | 0.9500 |

15 条可回答题中，9 条正确证据排第 1、3 条排第 2、2 条排第 3。唯一失败
题被相邻主题的多个 Chunk 干扰，正确证据排第 6。该失败不再用于调整
Embedding、阈值、切分参数或查询规则。

## 运行

```bash
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --questions data/eval/retrieval_v1.json \
  --retriever bm25 \
  --chunk-size 1200 \
  --chunk-overlap 150 \
  --output evals/reports/retrieval_v1_bm25.json
```

```bash
FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --questions data/eval/retrieval_v1.json \
  --retriever vector \
  --embedding-cache var/embeddings.json \
  --chunk-size 1200 \
  --chunk-overlap 150 \
  --vector-min-score 0.55 \
  --output evals/reports/retrieval_v1_vector.json
```

私有留出集使用固定配置运行：

```bash
FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uv run firmware-rag-eval \
  --catalog var/private-corpus/sources.json \
  --questions var/private-corpus/holdout-v1.json \
  --retriever vector \
  --embedding-cache var/private-embeddings.json \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --vector-min-score 0.55 \
  --vector-store var/vector-store/private-wiki \
  --vector-collection private_firmware_knowledge \
  --strict-label-audit \
  --output var/private-corpus/holdout-v1-vector-raw.json
```
