# Multimodal Asset RAG

一个最小但完整的多模态资料检索项目，用于 LlamaIndex 进阶教程。

## 能力

- PDF 解析：默认 PyMuPDF，可切 PaddleOCR-VL API
- 图片解析：可选本地 PaddleOCR HTTP 和 OpenAI-compatible VLM caption
- 向量库：默认 Qdrant Local，可切 Qdrant Server
- 文本向量：OpenAI-compatible embedding 服务
- 图片向量：默认 lite 视觉特征，可选本地 `sentence_transformers` CLIP
- 检索：文本检索、文搜图、图搜图、混合检索
- 生成：OpenAI-compatible LLM 基于证据回答
- 评估：固定 query 的最小命中率检查

## 架构

```text
API / CLI
  -> Services(parse, index, search, answer, eval)
  -> Providers(PaddleOCR-VL, OCR HTTP, VLM, Embedding, CLIP)
  -> Repositories(Qdrant Local/Server, parsed artifacts)
  -> Assets(PDF, images, manifest)
```

第一版不引入知识图谱。当前问题主要是多模态解析、向量召回、证据溯源和排序融合，知识图谱会增加建模成本，但不能直接解决图片/PDF 解析质量。等后面出现实体关系类问题，比如“模型、论文、作者、方法之间的关系追踪”，再单独加 graph 层。

Rerank 也不是第一步必需。建议先把召回做好：PDF 文本、VLM caption、CLIP 图向量各自能命中，再用评估集判断是否需要 rerank。项目已经预留配置位，后续可以接你的 rerank 模型。

## 运行

```bash
cd llamaindex/code
python -m multimodal_asset_rag.cli parse --limit 5
python -m multimodal_asset_rag.cli index
python -m multimodal_asset_rag.cli search "哪份资料讲了 retrieval augmented generation？"
python -m multimodal_asset_rag.cli answer "哪份资料讲了 retrieval augmented generation？"
python -m multimodal_asset_rag.cli eval
```

默认 `index` 使用 Qdrant Local，数据保存在：

```text
llamaindex/data/media_project/multimodal_asset_rag/indexes/qdrant
```

启动 API：

```bash
uvicorn multimodal_asset_rag.api:app --host 127.0.0.1 --port 8011
```

标准接口：

```text
GET  /health
POST /ingest
POST /search
POST /answer
POST /eval
```

如果使用 PaddleOCR-VL：

```bash
python -m multimodal_asset_rag.cli parse --pdf-parser paddleocr_vl --limit 2
```

如果使用图片 OCR / VLM：

```bash
python -m multimodal_asset_rag.cli parse --ocr --vlm --limit 10
```
