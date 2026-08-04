# Firmware Knowledge Agent

面向嵌入式固件文档的 **RAG**：有证据就带引用回答，没证据就拒答。

这个项目的重点不是"跑通 RAG"，而是**用可复现的评测回答检索到底有没有变好**。
每一个结论都标注了它成立的边界。

---

## 30 秒速览

| 维度 | 现状 |
| --- | --- |
| 检索 | BM25 / 向量 / RRF 混合三条链路，可选 ONNX CrossEncoder 重排 |
| 向量 | Gemini `gemini-embedding-001` 或本地 Ollama `qwen3-embedding:0.6b` |
| 存储 | Qdrant 本地持久化，语料或 Embedding 模型变化时自动重建 |
| 入库 | Markdown / TXT / HTML / PDF / DOCX，含清洗、脱敏、增量更新 |
| 编排 | LangGraph：检索 → 证据门控 → 生成 → 引用校验 → 完成 / 降级 / 拒答 |
| 拒答 | 向量相似度门槛 `0.55`，无语义证据时不返回结果 |
| 测试 | 43 项自动化测试通过 |

```text
retrieve -> grade_evidence
              ├── 有证据 -> generate -> verify_citations ─┬─> complete
              │                                           └─> fallback_extractive
              └── 无证据 -> refuse
```

生成后校验引用编号：模型没给引用、编号越界或调用失败时，**降级为抽取式答案**
而不是把无依据的回答返回给用户。无证据时根本不调用生成器。

---

## 评测结果

### 公开语料（ESP-IDF / FreeRTOS 官方文档，可复现）

50 题 v1 开发集：45 题可答 + 5 题语料外。`top_k=3`、chunk `1200/150`、
Embedding `qwen3-embedding:0.6b`、门槛 `0.55`。

| 检索器 | Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4889 | 0.4370 | 0.8000 | 0.5200 |
| Vector | 0.9556 | 0.8407 | 1.0000 | 0.9600 |
| Hybrid RRF | 0.9556 | 0.8148 | 1.0000 | 0.9600 |
| Vector + 领域查询规范化 | 1.0000 | 0.8593 | 1.0000 | 1.0000 |

**参数冻结后**新建 18 题独立留出集（15 可答 + 3 语料外）：

| Hit@3 | MRR | 拒答准确率 |
| ---: | ---: | ---: |
| 0.8667 | 0.8000 | 1.0000 |

### 私有业务语料（31 篇脱敏文档 / 316 个 Chunk，不随项目分发）

参数冻结后的 20 题留出集（15 可答 + 5 条业务相邻但语料未覆盖的困难拒答题）：

| Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| ---: | ---: | ---: | ---: |
| 0.9333 | 0.7444 | 1.0000 | 0.9500 |

15 条可答题中 9 条证据排第 1、3 条排第 2、2 条排第 3。唯一失败题的正确证据被
相邻主题干扰排到第 6，该失败**不再用于调整任何参数**。

---

## 评测方法：这些数字为什么可信

这一节是项目的核心。RAG 很容易做出好看但没有意义的数字。

### 命中判定不看相似度，看证据词

只有 Top-K 段落**来自正确来源**、并且**包含每组至少一个人工标注的证据词组**
才算命中。语料外问题只有返回 `no_evidence` 才算通过。

`--strict-label-audit` 会在检索前校验：每道可答题必须在正确来源的**同一个
Chunk** 中包含全部证据词组，否则评测直接终止。这防止标注本身就是错的。

### 开发集和留出集严格分离

阈值 `0.55` 是从 v1 数据集的分数分布推出来的：可答题最高分的**最低值是
0.619**，语料外题最高分的**最高值是 0.451**，所以门槛取中间。

这是**在开发集上探索得到的**，因此不能把 v1 的 96% 表述成泛化准确率。留出集
在参数冻结后才建立，且建立后不再回头调参——否则留出集会被污染成新的开发集。

降低门槛的对照也做了：`0.55` → `0.45` 时 Hit@3 和 MRR 没有提升，域外拒答准确率
反而从 `1.0000` 掉到 `0.8000`，所以保留更保守的值。

### 失败原样保留，不删题不放宽标签

v1 中两条失败题（`freertos-task-name-purpose` 证据跨 chunk 边界、
`nvs-power-loss-guarantee` Top-3 未命中断电保证）的正确证据原本在向量候选第 5
和第 8 位。项目增加了两条**可解释的**领域查询规范化规则后回归满分，优化前的
报告仍然保留用于对比。

这只能证明已知失败被修复，**不能表述成未知问题上 100%**。

留出集中两条 Wi-Fi 扫描题至今未命中，原样保留。

### 标注错误也记录

18 题留出集首次运行的原始报告保存在 `holdout_v1_vector_raw.json`。后续审计发现
两处人工标注问题：一处证据短语被 Markdown 换行拆开，另一处把文档中的
`Synchronization events` 错标为 `external`。修正标签后重算，检索参数和查询规则
没有动。原始与修正后的报告都保留。

### 结果可复现

评测报告的 `run_config` 记录检索器、Reranker、Embedding 模型、切分参数、
相似度门槛，以及**题集和语料目录的 SHA256**。

### Reranker 没有被包装成"必然更好"

私有语料上的对照：

| 检索链路 | Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| --- | ---: | ---: | ---: | ---: |
| Vector | 1.0000 | **0.9624** | 1.0000 | 1.0000 |
| Hybrid RRF | 1.0000 | 0.8978 | 1.0000 | 1.0000 |
| Hybrid + ONNX CrossEncoder | 1.0000 | 0.9462 | 1.0000 | 1.0000 |

Reranker 确实把 Hybrid 的 MRR 从 `0.8978` 提到 `0.9462`，但**仍低于纯向量的
`0.9624`**。所以默认链路是 Vector，Hybrid 和 Reranker 作为可切换的对照保留，
而不是因为技术名词好听就默认开启。

完整口径、失败分析和逐条结果见 [`evals/README.md`](evals/README.md) 与
`evals/reports/`。

---

## 快速开始

```bash
uv sync --extra dev
uv run firmware-rag-ingest-public
uv run firmware-rag search "周期任务应该使用哪个延时 API？"
uv run firmware-rag answer "NVS 写入后为什么要调用 commit？"
```

`firmware-rag-ingest-public` 读取 `data/public_sources.json`，只允许访问配置中
的官方 HTTPS 域名，结果写入被 Git 忽略的 `var/public-corpus`。

跑一次评测：

```bash
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --questions data/eval/retrieval_v1.json \
  --retriever bm25 \
  --chunk-size 1200 --chunk-overlap 150
```

换本地向量检索（无需 API Key）：

```bash
ollama pull qwen3-embedding:0.6b

FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --questions data/eval/retrieval_v1.json \
  --retriever vector \
  --chunk-size 1200 --chunk-overlap 150 \
  --vector-min-score 0.55
```

启动服务：

```bash
FIRMWARE_RAG_CATALOG=var/public-corpus/sources.json \
FIRMWARE_RAG_RETRIEVER=vector \
FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
FIRMWARE_RAG_CHUNK_SIZE=1200 \
FIRMWARE_RAG_CHUNK_OVERLAP=150 \
FIRMWARE_RAG_VECTOR_MIN_SCORE=0.55 \
  uv run uvicorn firmware_knowledge_agent.api:app --port 8010
```

Swagger：`http://127.0.0.1:8010/docs`

---

## API

```bash
curl -X POST http://127.0.0.1:8010/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Wi-Fi 断开后如何重连？","top_k":3}'
```

`/v1/answer` 返回带引用的抽取式基线，`/v1/agent/answer` 走完整的 LangGraph
生成与校验链路：

```bash
curl -X POST http://127.0.0.1:8010/v1/agent/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"NVS 写入后为什么需要 nvs_commit？","top_k":3}'
```

默认生成器是可离线运行的 `extractive`；设置 `FIRMWARE_RAG_GENERATOR=ollama`
后使用本机 `llama3.1:8b`。

上传文档会清洗文本、脱敏、写入私有语料目录并触发重新加载，单文件上限 10 MiB：

```bash
curl -X POST http://127.0.0.1:8010/v1/corpus/upload \
  -F 'file=@/absolute/path/to/document.pdf' \
  -F 'title=设备故障排查说明'
```

也可以绕过 HTTP 直接本地导入：

```bash
uv run firmware-rag-ingest-local /absolute/path/to/document.docx \
  --catalog var/private-corpus/sources.json
```

---

## 语料分层

| 目录 | 内容 | 是否提交 |
| --- | --- | --- |
| `data/sample/` | 3 份公开文档摘要，仅用于验证代码和测试 | 是 |
| `data/eval/` | 6 题 pilot、50 题开发集、18 题留出集 | 是 |
| `var/public-corpus/` | 采集的 ESP-IDF / FreeRTOS 官方文档 | 否 |
| `var/private-corpus/` | 31 篇脱敏业务文档 + 私有题集与报告 | 否 |
| `var/vector-store/` | Qdrant 向量索引 | 否 |

私有语料保存两层：`var/private-corpus/` 存脱敏后的 Markdown 与来源目录（便于
重新切分、更新和审计），`var/vector-store/` 存 Qdrant 索引。两者都在 Git 忽略
的 `var/` 下——**Git 忽略只表示不提交，不影响服务读取或建索引**。私有语料仅限
本机演示，不随项目分发。

---

## 验证

```bash
uv run pytest -q     # 43 passed
uv run python -m compileall -q src tests
```

---

## 目录

```text
src/firmware_knowledge_agent/
  corpus.py            # 目录清单、文档读取和切分
  local_ingestion.py   # 本地文件解析、清洗、脱敏、增量更新
  public_ingestion.py  # 官方网页采集与 catalog 生成
  embeddings.py        # Gemini / Ollama Embedding 与本地缓存
  retrieval.py         # 查询规范化、BM25、向量与混合检索
  reranking.py         # ONNX CrossEncoder 重排
  vector_store.py      # Qdrant 持久化索引与语料指纹
  service.py           # 搜索、引用和拒答
  agentic_workflow.py  # LangGraph 证据门控、生成、引用守卫和降级
  evaluation.py        # Hit@K / MRR / 拒答准确率
  api.py               # FastAPI
evals/
  README.md            # 评测设计、口径和失败分析
  reports/             # BM25 / Vector / Hybrid JSON 报告
```

---

## 下一阶段

1. 建立生成答案评测，检查引用完整性和事实是否被证据支持（当前只评测了检索）。
2. 增加上传权限、异步索引任务和语料版本管理，再考虑多人使用。
3. 扩大私有语料规模，验证当前结论在更大语料上是否成立。

逆向学习顺序见 [`docs/PROJECT_WALKTHROUGH.md`](docs/PROJECT_WALKTHROUGH.md)。
