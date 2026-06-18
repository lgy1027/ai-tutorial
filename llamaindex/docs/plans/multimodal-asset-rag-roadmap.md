# Multimodal Asset RAG 项目推进计划

## 目标

我们接下来不再先写文章，而是先把一个完整、可独立部署、可开源的小型多模态 RAG 项目做出来。

项目目标是：

> 面向真实 PDF 和图片资料，完成解析、入库、检索、回答、溯源、评估和调优闭环。

它不是 OCR Demo，也不是单纯的 LlamaIndex API 示例。它应该像一个小型开源项目，有清晰架构、标准接口、可运行 CLI、可部署 API、真实模型服务、真实向量库和可复现评估。

## 当前已有基础

当前仓库里已经有一个雏形项目：

```text
llamaindex/code/multimodal_asset_rag/
```

已经具备：

- PDF 解析：默认 PyMuPDF，预留 PaddleOCR-VL API
- 图片解析：预留 OCR HTTP 和 VLM caption
- 文本索引：已支持 LlamaIndex storage，正在升级到 Qdrant
- 图片索引：已有 lite 兜底，预留本地 CLIP
- 检索：支持 text、text-to-image、image-to-image、hybrid
- 回答：支持 OpenAI-compatible LLM 基于证据回答
- 评估：已有最小 eval cases
- API：已有 FastAPI 雏形

已有真实素材：

```text
llamaindex/data/media_project/chapter11_assets/
```

包含：

- 10 个 AI 相关 PDF
- 20+ 张真实图片
- asset manifest

## 架构方向

项目按四层组织：

```text
API / CLI
  -> Services
  -> Providers
  -> Repositories
  -> Assets
```

### API / CLI

提供两种入口：

- CLI：本地开发、教程演示、批处理
- FastAPI：独立部署、前端调用、服务化集成

标准接口计划：

```text
GET  /health
POST /ingest
POST /search
POST /answer
POST /eval
```

### Services

服务层负责业务流程编排：

- parse service：PDF / 图片解析
- index service：构建文本向量和图片向量
- search service：文本检索、文搜图、图搜图、混合检索
- answer service：基于检索证据生成答案
- eval service：固定问题集评估

### Providers

模型和外部能力统一封装成 provider：

- OCR provider：PaddleOCR-VL 官方 API
- OCR fallback：本地 PaddleOCR HTTP
- VLM provider：MiniMax-M3 OpenAI-compatible
- Embedding provider：OpenAI-compatible embedding
- Image embedding provider：本地 CLIP / SigLIP
- Rerank provider：先预留，后续接入

### Repositories

存储层统一管理：

- parsed artifacts：Markdown、OCR JSON、caption JSON、raw JSONL
- vector store：Qdrant Local / Qdrant Server
- asset manifest：素材清单
- eval report：评估结果

## 技术选型

### 向量库

第一版推荐使用 Qdrant。

原因：

- Qdrant Local 可以本地持久化，不需要先启动 Docker
- 后续可以平滑切换到 Qdrant Server
- Python 客户端清晰，适合开源项目和教程
- 支持 payload metadata，方便做溯源和过滤

Milvus Lite 也可以，但第一版先不引入，减少部署复杂度。

### 知识图谱

第一版暂时不做知识图谱。

原因：

- 当前核心问题是多模态解析、向量检索、图文检索、证据溯源和调优
- 知识图谱更适合实体关系问题，比如论文、作者、模型、方法之间的关系追踪
- 现在加入 graph 层会提高复杂度，但不能直接解决 PDF/图片检索不准的问题

后续如果系列进入“论文知识图谱”或“方法关系追踪”，再单独做 graph layer。

### Rerank

Rerank 需要预留，但不作为第一阶段硬依赖。

原因：

- 先要确认召回层能命中正确候选
- 如果召回不到，rerank 没有意义
- 等 text / image / hybrid recall 稳定后，再接 rerank 做排序优化

如果需要接入 rerank，计划使用和 embedding 共用的请求地址和 API key，只额外配置 `RERANK_MODEL`。

## 阶段计划

### 阶段一：工程化项目骨架

目标：把当前雏形整理成一个独立开源小项目。

要做：

- 整理目录结构
- 明确 CLI 命令
- 明确 FastAPI 接口 schema
- 整理配置项
- 补充 README
- 统一日志和错误处理
- 明确生成物目录，避免污染 git

验收标准：

```bash
cd llamaindex/code
python -m multimodal_asset_rag.cli parse --limit 5
python -m multimodal_asset_rag.cli index
python -m multimodal_asset_rag.cli search "哪份资料讲了 retrieval augmented generation？"
python -m multimodal_asset_rag.cli answer "哪份资料讲了 retrieval augmented generation？"
python -m multimodal_asset_rag.cli eval
```

API 能启动：

```bash
uvicorn multimodal_asset_rag.api:app --host 127.0.0.1 --port 8011
```

### 阶段二：接入真实模型服务

目标：不用 mock，全部走真实服务。

要做：

- OCR：默认接 PaddleOCR-VL 官方 API
- VLM：接 MiniMax-M3 OpenAI-compatible
- Embedding：接已有 OpenAI-compatible embedding 服务
- CLIP：本地部署最小 CLIP / SigLIP
- Qdrant：使用 Qdrant Local 持久化

注意事项：

- embedding 服务需要做 batch、节流、重试
- PaddleOCR-VL 解析结果要缓存，避免重复消耗 API
- 本地 CLIP 首次加载要记录模型名和向量维度
- Qdrant collection 名称要包含向量维度，避免模型切换后污染

验收标准：

- PDF 能通过 PaddleOCR-VL 生成 Markdown
- 图片能生成 VLM caption
- 文本向量进入 Qdrant
- 图片向量进入 Qdrant
- text search、text-to-image、image-to-image、hybrid search 均可运行

### 阶段三：标准化检索接口

目标：接口不暴露内部实现细节，适合前端或外部服务调用。

要做：

- `/search` 支持：
  - text
  - text-to-image
  - image-to-image
  - hybrid
- `/answer` 返回：
  - answer
  - sources
  - routes
  - score
  - asset_id
  - source_path
  - page
  - parser
- 支持 metadata filter：
  - source_type
  - tags
  - parser
  - asset_id

验收标准：

搜索结果必须能回答：

- 命中了哪个文件？
- PDF 是第几页？
- 图片原图在哪里？
- 是哪条检索路线命中的？
- 使用了哪个解析器？

### 阶段四：调优体系

目标：把“搜不准”拆成可定位的问题。

调优分层：

```text
解析质量
  -> 资产组织
  -> 单路召回
  -> 多路融合
  -> rerank
  -> 评估集
```

要做：

- 建立 eval cases
- 每条 case 标注 expected asset_id
- 记录 top-k 命中情况
- 记录错误类型：
  - parse_error
  - missing_caption
  - embedding_miss
  - clip_visual_mismatch
  - fusion_bad_weight
  - rerank_error
- 输出 eval report

验收标准：

- 每次修改模型、chunk、caption、CLIP、fusion 权重后，可以跑同一批 eval cases
- 能看出是召回问题还是排序问题

### 阶段五：公众号文章反写

目标：基于真实项目写文章，而不是先写概念。

计划拆成多期：

1. 第一篇：为什么多模态 RAG 不能只是 OCR Demo，从资料资产流水线开始
2. 第二篇：PaddleOCR-VL 解析 PDF 和图片，如何保留 Markdown、页码和证据
3. 第三篇：VLM caption 和 OCR 怎么配合，为什么图片不能只转文字
4. 第四篇：CLIP / SigLIP 做文搜图和图搜图，为什么会搜不准
5. 第五篇：Qdrant 多集合检索，文本向量和图片向量怎么组织
6. 第六篇：hybrid search、rerank 和评估集，系统性优化多模态检索

## 需要用户提供的信息

目前还需要确认或提供：

- Rerank 模型名，如果要接入 rerank
- 本地 CLIP / SigLIP 倾向使用哪个模型
- PaddleOCR-VL API 是否作为默认 PDF 解析路线
- Qdrant 是否只用 Local，还是后续也要提供 Docker Compose
- 是否需要一个简单前端页面展示搜索结果和图片

## 当前不做

第一版暂不做：

- 知识图谱
- 多用户权限
- 大规模任务队列
- 前端复杂交互
- 自动标注训练集
- 微调 embedding / rerank

这些都可以后续加，但第一版先把多模态 RAG 的工程主链路跑稳。

