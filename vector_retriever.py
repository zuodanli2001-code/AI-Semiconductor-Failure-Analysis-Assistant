
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_retriever import load_documents


# ==================== 基础配置 ====================

BASE_DIR = Path(__file__).resolve().parent

# 使用轻量级中文语义向量模型
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 向量缓存文件，首次生成后会保存在项目目录中
CACHE_FILE = BASE_DIR / "vector_index.npz"

_model = None


# ==================== 加载向量模型 ====================

def get_model():
    """首次调用时加载模型，后续重复使用。"""
    global _model

    if _model is None:
        print(f"正在加载向量模型：{MODEL_NAME}")
        print("首次运行可能需要下载模型，请耐心等待。")

        _model = SentenceTransformer(MODEL_NAME)

        print("向量模型加载完成。")

    return _model


# ==================== 生成文档签名 ====================

def get_document_signature(documents):
    """根据文档内容生成签名，用于判断缓存是否需要更新。"""
    content = "\n".join(
        f"{doc.get('source', '')}|"
        f"{doc.get('page', '')}|"
        f"{doc.get('chunk_id', '')}|"
        f"{doc.get('content', '')}"
        for doc in documents
    )

    return hashlib.md5(content.encode("utf-8")).hexdigest()


# ==================== 建立或读取向量索引 ====================

def build_or_load_index(documents, model):
    """优先加载有效缓存；文档变化后重新生成向量。"""
    signature = get_document_signature(documents)

    if CACHE_FILE.exists():
        try:
            with np.load(CACHE_FILE, allow_pickle=False) as cache:
                cached_signature = str(cache["signature"].item())
                embeddings = cache["embeddings"]

            if (
                cached_signature == signature
                and len(embeddings) == len(documents)
            ):
                print("已加载本地向量缓存。")
                return embeddings

            print("检测到知识库发生变化，将重新生成向量。")

        except (ValueError, KeyError, OSError):
            print("原有向量缓存不可用，将重新生成。")

    print(f"正在为 {len(documents)} 个文本片段生成向量……")

    texts = [str(doc.get("content", "")) for doc in documents]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.savez_compressed(
        CACHE_FILE,
        signature=np.array(signature),
        embeddings=embeddings,
    )

    print(f"向量已保存至：{CACHE_FILE}")

    return embeddings


# ==================== 向量检索 ====================


def search_knowledge_vector(query, top_k=3):
    """根据语义相似度检索知识库，严格返回分数最高的 Top-K 结果。"""
    query = query.strip()

    if not query:
        return []

    documents = load_documents()

    # 排除没有正文的片段
    documents = [
        doc for doc in documents
        if str(doc.get("content", "")).strip()
    ]

    if not documents:
        print("知识库中没有可检索的文档。")
        return []

    model = get_model()
    document_embeddings = build_or_load_index(documents, model)

    # 为查询生成向量
    query_text = (
        "为这个句子生成表示以用于检索相关文章：" + query
    )

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # 计算余弦相似度
    scores = document_embeddings @ query_embedding

    # 按相似度从高到低排序
    ranked_indices = np.argsort(scores)[::-1]

    # 防止 top_k 不合理
    limit = max(1, int(top_k))
    limit = min(limit, len(documents))

    results = []

    for index in ranked_indices[:limit]:
        index = int(index)
        document = documents[index]

        results.append({
            **document,
            "score": round(float(scores[index]), 4),
            "matched_keywords": [],
        })

    return results
