# Firmware Knowledge Agent

面向嵌入式开发文档和故障排查场景的 RAG 项目。

项目完成后的逆向学习顺序见
[`docs/PROJECT_WALKTHROUGH.md`](docs/PROJECT_WALKTHROUGH.md)。

当前版本是一条可运行、可评测的本地 RAG 链路，重点回答五个问题：

1. 文档能否按章节切分并保留来源元数据？
2. 给定问题能否检索到包含答案证据的正确段落？
3. 没有证据时能否拒绝回答？
4. 关键词、向量和混合检索在同一数据集上的差异是什么？
5. Reranker 是否真的改善当前语料，而不是只增加技术名词？

`data/eval/baseline.json` 的 6 题 pilot 只用于冒烟测试。
`data/eval/retrieval_v1.json` 包含 50 条人工检查的问题：45 条可回答题和
5 条语料外问题，用于同时评测检索与拒答。

## 当前能力

- Markdown 标题感知切分和二次长度切分。
- RST 标题转 Markdown，保留 ESP-IDF 官方文档章节结构。
- 固件领域查询规范化，把自然语言扩展为对应 API 和事件标识符。
- BM25 关键词检索。
- Gemini `gemini-embedding-001` 向量检索。
- Ollama `qwen3-embedding:0.6b` 本地多语言向量检索。
- BM25 + 向量检索的 RRF 混合检索。
- Qdrant 本地持久化向量数据库，语料或 Embedding 模型变化时自动重建。
- Markdown、TXT、HTML、PDF、DOCX 本地导入和 FastAPI 文件上传。
- 导入文本清洗、常见敏感字段脱敏、来源目录增量更新。
- 本地文档向量缓存，避免重复调用 Embedding API。
- 可选 ONNX CrossEncoder Reranker，并保留无重排基线。
- 来源 URL、文档标题、章节和 chunk ID 引用。
- 无证据兜底。
- 基于来源和答案证据词的 Hit@K、MRR、拒答准确率和总体准确率评测。
- 可配置向量相似度门槛，混合检索在没有语义证据时拒绝返回结果。
- FastAPI 搜索与回答接口。
- LangGraph `检索 -> 证据检查 -> 生成或拒答` 工作流。
- 抽取式与本机 Ollama 两种回答生成器。
- 生成后引用编号校验；缺少合法引用或模型失败时降级为抽取式答案。

`data/sample` 只有三份自行整理的公开文档摘要，用于验证代码和测试，不能作为简历项目的最终语料规模。

## 运行

```bash
uv sync --extra dev
uv run firmware-rag-ingest-public
uv run firmware-rag search "周期任务应该使用哪个延时 API？"
uv run firmware-rag answer "NVS 写入后为什么要调用 commit？"
FIRMWARE_RAG_GENERATOR=ollama \
  uv run firmware-rag agent-answer "Wi-Fi 断开后如何重连？"
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --questions data/eval/retrieval_v1.json \
  --retriever bm25 \
  --chunk-size 1200 \
  --chunk-overlap 150
uv run uvicorn firmware_knowledge_agent.api:app --reload --port 8010
```

`firmware-rag-ingest-public` 读取 `data/public_sources.json`，只允许访问配置中的官方 HTTPS 域名，并把采集结果写到被 Git 忽略的 `var/public-corpus`。使用正式语料时：

```bash
FIRMWARE_RAG_CATALOG=var/public-corpus/sources.json \
FIRMWARE_RAG_VECTOR_STORE=var/vector-store/public \
  uv run uvicorn firmware_knowledge_agent.api:app --reload --port 8010
```

私有业务语料使用两层本地存储：

1. `var/private-corpus/` 保存经过筛选和脱敏的 Markdown 语料与来源目录，便于重新切分、更新和审计。
2. `var/vector-store/` 保存 Qdrant 向量索引，用于实际语义检索。

两者都位于 Git 忽略的 `var/` 下。Git 忽略只表示不会提交到代码仓库，
不影响服务读取或建立 RAG 索引。私有语料仅限本机演示，不随项目分发。

```bash
FIRMWARE_RAG_CATALOG=var/private-corpus/sources.json \
FIRMWARE_RAG_RETRIEVER=vector \
FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
FIRMWARE_RAG_VECTOR_STORE=var/vector-store/private-wiki \
FIRMWARE_RAG_CHUNK_SIZE=1000 \
FIRMWARE_RAG_CHUNK_OVERLAP=150 \
  uv run uvicorn firmware_knowledge_agent.api:app --port 8010
```

私有语料的检索冒烟评测同样保存在 `var/`，不会进入代码仓库：

```bash
FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
  uv run firmware-rag-eval \
  --catalog var/private-corpus/sources.json \
  --questions var/private-corpus/holdout-v1.json \
  --retriever vector \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --vector-min-score 0.55 \
  --vector-store var/vector-store/private-wiki \
  --vector-collection private_firmware_knowledge \
  --strict-label-audit
```

当前私有语料包含 31 篇经过筛选和脱敏的业务技术文档。回归集共 36 题：
31 题可回答、5 题域外拒答。

| 检索链路 | Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| --- | ---: | ---: | ---: | ---: |
| Ollama Vector | 1.0000 | 0.9624 | 1.0000 | 1.0000 |
| BM25 + Vector RRF | 1.0000 | 0.8978 | 1.0000 | 1.0000 |
| Hybrid + ONNX CrossEncoder | 1.0000 | 0.9462 | 1.0000 | 1.0000 |

Reranker 将 Hybrid 的 MRR 从 `0.8978` 提升到 `0.9462`，但仍低于纯向量
的 `0.9624`，因此当前演示默认使用 Vector，Hybrid 和 Reranker 作为可切换
对照链路保留。这组题用于验证真实语料链路和建立回归基线，仍与语料同源，
不能表述成未知问题上的通用准确率。

向量门槛从 `0.55` 降到 `0.45` 时，Hit@3 和 MRR 没有提升，域外拒答
准确率反而从 `1.0000` 降至 `0.8000`，因此保留更保守的 `0.55`。

参数冻结后另建 20 题私有留出集，其中 15 题可回答、5 题为业务相邻但
语料未覆盖的问题。首次运行结果为 `Hit@3=0.9333`、`MRR=0.7444`、
拒答准确率 `1.0000`、总体准确率 `0.9500`。唯一失败题的正确证据位于
第 6 名，失败原样保留，不再用该留出集调整阈值或查询规则。报告同时记录
题集和语料 SHA256、检索参数与 Embedding 模型，便于确认结果可复现。

安装可选依赖后可复现 Reranker 对照实验：

```bash
uv sync --extra dev --extra rerank

FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
  uv run firmware-rag-eval \
  --catalog var/private-corpus/sources.json \
  --questions var/private-corpus/eval.json \
  --retriever hybrid \
  --reranker cross-encoder \
  --reranker-cache var/models \
  --reranker-candidates 12 \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --vector-store var/vector-store/private-wiki \
  --vector-collection private_firmware_knowledge
```

Swagger：

```text
http://127.0.0.1:8010/docs
```

## API

```bash
curl -X POST http://127.0.0.1:8010/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Wi-Fi 断开后如何重连？","top_k":3}'
```

```bash
curl -X POST http://127.0.0.1:8010/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"NVS 如何保存配置？","top_k":3}'
```

上传本地文档后会清洗文本、写入被 Git 忽略的私有语料目录，并触发服务
重新加载。支持 `.md`、`.markdown`、`.txt`、`.html`、`.htm`、`.pdf`
和 `.docx`，单文件上限 10 MiB：

```bash
curl -X POST http://127.0.0.1:8010/v1/corpus/upload \
  -F 'file=@/absolute/path/to/document.pdf' \
  -F 'title=设备故障排查说明'
```

也可以不经过 HTTP，直接使用本地导入命令：

```bash
uv run firmware-rag-ingest-local \
  /absolute/path/to/document.docx \
  --catalog var/private-corpus/sources.json
```

`/v1/answer` 返回带引用的抽取式基线；`/v1/agent/answer` 则进入下面的
LangGraph 生成与校验链路。

`/v1/agent/answer` 使用 LangGraph 显式执行：

```text
retrieve
-> grade_evidence
-> generate -> verify_citations -> complete / fallback_extractive
-> refuse
```

默认生成器仍是可离线运行的 `extractive`。设置
`FIRMWARE_RAG_GENERATOR=ollama` 后使用本机 `llama3.1:8b`。无证据时不会
调用生成器；模型无引用、引用编号越界或调用失败时退回抽取式答案。

```bash
curl -X POST http://127.0.0.1:8010/v1/agent/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"NVS 写入后为什么需要 nvs_commit？","top_k":3}'
```

正式英文语料使用中文提问时，可使用同一份评测集比较三种检索器：

```bash
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --retriever bm25 \
  --chunk-size 1200 \
  --chunk-overlap 150

FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --retriever vector \
  --chunk-size 1200 \
  --chunk-overlap 150

FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --retriever hybrid \
  --chunk-size 1200 \
  --chunk-overlap 150
```

50 题 v1 基线（`top_k=3`、chunk `1200/150`）：

| 检索器 | Hit@3 | MRR | 拒答准确率 | 总体准确率 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4889 | 0.4370 | 0.8000 | 0.5200 |
| Ollama Vector | 0.9556 | 0.8407 | 1.0000 | 0.9600 |
| BM25 + Vector RRF | 0.9556 | 0.8148 | 1.0000 | 0.9600 |
| Vector + 失败驱动查询规范化 | 1.0000 | 0.8593 | 1.0000 | 1.0000 |

向量与混合检索使用 `qwen3-embedding:0.6b`，相似度门槛暂定为 `0.55`。
这个门槛由同一 v1 数据集的分数分布探索得到，还没有独立留出集验证，
不能外推成未知问题上的准确率。完整报告见 `evals/reports/`。

当前保留两条真实失败：`pcName` 的答案跨 chunk 边界，以及 NVS 断电问题
没有在 Top-3 命中正确证据。由于向量检索的 MRR 高于 RRF，现阶段不把
混合检索包装成“必然更好”。

随后针对这两条失败增加固件领域查询规范化，在同一 v1 集上回归为满分。
这是基于已知失败的优化结果，不是独立测试集，不能写成未知问题准确率
100%。优化前后的 JSON 报告都保留在 `evals/reports/`。

参数固定后建立的 18 题留出集包含 15 条可回答题和 3 条语料外问题：
`Hit@3=0.8667`、`MRR=0.8000`、拒答准确率 `1.0000`。两条 Wi-Fi
扫描问题未命中并原样保留。首次原始报告中另有两处人工证据标签错误，
原始与修正后报告均保留，具体见 `evals/README.md`。

Gemini 免费额度不足时可以使用本机 Ollama：

```bash
ollama pull qwen3-embedding:0.6b

FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uv run firmware-rag-eval \
  --catalog var/public-corpus/sources.json \
  --retriever vector \
  --chunk-size 1200 \
  --chunk-overlap 150
```

启动当前官方语料 + 本地混合检索配置：

```bash
FIRMWARE_RAG_CATALOG=var/public-corpus/sources.json \
FIRMWARE_RAG_RETRIEVER=hybrid \
FIRMWARE_RAG_EMBEDDING_PROVIDER=ollama \
FIRMWARE_RAG_EMBEDDING_MODEL=qwen3-embedding:0.6b \
FIRMWARE_RAG_CHUNK_SIZE=1200 \
FIRMWARE_RAG_CHUNK_OVERLAP=150 \
FIRMWARE_RAG_VECTOR_MIN_SCORE=0.55 \
FIRMWARE_RAG_GENERATOR=ollama \
FIRMWARE_RAG_OLLAMA_MODEL=llama3.1:8b \
uv run uvicorn firmware_knowledge_agent.api:app --port 8010
```

## 目录

```text
data/
  sample/              # 工程验证样例，不是最终语料
  eval/                # 6 题 pilot、50 题开发集和 18 题留出集
  public_sources.json  # 允许采集的官方页面清单
evals/
  README.md            # 评测设计、口径和失败分析
  reports/             # BM25、Vector、Hybrid JSON 报告
src/firmware_knowledge_agent/
  corpus.py            # 目录清单、文档读取和切分
  local_ingestion.py   # 本地文件解析、清洗、脱敏和目录增量更新
  public_ingestion.py  # 官方网页采集与本地 catalog 生成
  embeddings.py        # Gemini/Ollama Embedding 与本地缓存
  retrieval.py         # 查询规范化、BM25、向量与混合检索
  reranking.py         # ONNX CrossEncoder 重排和候选包装
  vector_store.py      # Qdrant 本地持久化索引与语料指纹
  service.py           # 搜索、引用和拒答
  agentic_workflow.py  # LangGraph 证据门控、生成、引用守卫和降级
  evaluation.py        # Hit@K / MRR
  api.py               # FastAPI
tests/
```

## 验证

```bash
uv run pytest -q
uv run python -m compileall -q src tests
```

## 下一阶段

1. 建立私有语料独立留出集，验证当前检索结论是否能泛化。
2. 建立生成答案评测，检查引用完整性和事实是否被证据支持。
3. 增加上传权限、异步索引任务和语料版本管理，再考虑多人使用。
