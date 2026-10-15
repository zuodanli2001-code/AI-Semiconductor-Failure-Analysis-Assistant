import jieba
from rank_bm25 import BM25Okapi
from pathlib import Path
from collections import Counter
import json
import math
import re


# ==================== 1. 路径配置 ====================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"

# 新的 Markdown 文献目录
MARKDOWN_DIR = BASE_DIR / "literature" / "markdown"

# 旧 JSON 文献目录暂时保留，不再直接加载
LITERATURE_DIR = BASE_DIR / "literature" / "processed"


# 过滤常见的低价值中文词，减少无意义匹配
STOPWORDS = {
    "分析", "进行", "可以", "通过", "相关", "其中", "对于",
    "以及", "这种", "这些", "由于", "因此", "此外", "研究",
    "结果", "方法", "发现", "表明", "说明", "影响", "可能",
    "采用", "使用", "主要", "不同", "具有", "得到", "表现",
    "过程", "情况", "问题", "材料", "样品", "实验", "测试",
    "表征", "观察", "我们", "他们", "这个", "一个", "没有",
    "如果", "需要", "通常", "一般", "其中的", "可以通过",
    "the", "and", "for", "with", "from", "that", "this",
    "are", "was", "were", "into", "using", "used", "has",
    "have", "been", "which", "their", "these", "those",
}


# ==================== 2. 文本切分 ====================


import re




def split_text(text, max_length=1200, overlap=150):
    """优先按句子边界切片；限制片段长度，按完整句子保留适量重叠。"""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if len(text) <= max_length:
        return [text]

    # 按常见中英文句末标点拆分，并保留标点
    sentences = re.split(r"(?<=[。！？!?；;])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 只有单个句子超过上限时，才按字符兜底拆分
    units = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            units.append(sentence)
        else:
            for start in range(0, len(sentence), max_length):
                units.append(sentence[start:start + max_length])

    chunks = []
    current_units = []
    current_length = 0

    for unit in units:
        if current_units and current_length + len(unit) > max_length:
            chunks.append("".join(current_units))

            # 从上一片末尾选取完整句子作为重叠，
            # 重叠长度不超过 overlap，并确保能放下当前单元
            tail_units = []
            tail_length = 0

            for previous_unit in reversed(current_units):
                if tail_length + len(previous_unit) > overlap:
                    break
                if tail_length + len(previous_unit) + len(unit) > max_length:
                    break

                tail_units.insert(0, previous_unit)
                tail_length += len(previous_unit)

            current_units = tail_units
            current_length = tail_length

        # 如果当前单元仍然放不下，就不保留重叠
        if current_length + len(unit) > max_length:
            current_units = []
            current_length = 0

        current_units.append(unit)
        current_length += len(unit)

    if current_units:
        chunks.append("".join(current_units))

    return chunks



# ==================== 3. 加载 TXT 知识库 ====================

def load_txt_documents():
    """加载 knowledge_base 文件夹中的 TXT 文件。"""
    documents = []

    if not KNOWLEDGE_DIR.exists():
        return documents

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"TXT 读取失败：{file_path.name}，原因：{exc}")
            continue

        chunks = [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n+", text)
            if chunk.strip()
        ]

        for index, chunk in enumerate(chunks, start=1):
            documents.append({
                "source": file_path.name,
                "chunk_id": index,
                "content": chunk,
                "document_type": "txt",
            })

    return documents


# ==================== 4. 加载 PDF 文献 ====================


# ==================== 4. 加载 Markdown 文献 ====================

def load_markdown_documents():
    """读取 Markdown 文献，分别提取英文原文和中文翻译，保留页码。"""
    documents = []

    if not MARKDOWN_DIR.exists():
        print(f"Markdown 目录不存在：{MARKDOWN_DIR}")
        return documents

    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        # 翻译缓存是 JSON，不属于 Markdown；此处只读取 .md
        try:
            text = md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Markdown 读取失败：{md_path.name}，原因：{exc}")
            continue

        # 按每一页的标题切分
        page_sections = re.split(
            r"(?m)^## 第\s*(\d+)\s*页\s*$",
            text
        )

        # 正常结构：
        # page_sections[0] 是文档开头；
        # 后续依次是页码、该页正文、页码、该页正文……
        for index in range(1, len(page_sections), 2):
            page_number = int(page_sections[index])
            page_text = page_sections[index + 1].strip()

            # 提取英文原文和中文翻译
            english_match = re.search(
                r"(?s)### English Original\s*(.*?)"
                r"(?=### 中文翻译|$)",
                page_text
            )
            chinese_match = re.search(
                r"(?s)### 中文翻译\s*(.*?)(?=\n---\s*$|$)",
                page_text
            )

            english_text = (
                english_match.group(1).strip()
                if english_match else ""
            )
            chinese_text = (
                chinese_match.group(1).strip()
                if chinese_match else ""
            )

            # 每种语言独立入库，避免切片混淆语言
            for language, content in [
                ("en", english_text),
                ("zh", chinese_text),
            ]:
                if not content:
                    continue

                chunks = split_text(content)

                for chunk_index, chunk in enumerate(chunks, start=1):
                    documents.append({
                        "source": md_path.stem,
                        "page": page_number,
                        "chunk_id": f"{language}_{chunk_index}",
                        "content": chunk,
                        "language": language,
                        "document_type": "markdown",
                    })

    return documents


def load_documents():
    """同时加载 TXT 和 Markdown 文献，不重复加载旧 JSON。"""
    return load_txt_documents() + load_markdown_documents()


# ==================== 5. 关键词提取 ====================

def extract_keywords(text):
    """
    提取英文术语、数字和中文词组。
    中文连续文本同时生成二字组合，兼容不同问法。
    """
    tokens = re.findall(
        r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+",
        text.lower()
    )

    keywords = set()

    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) >= 2 and token not in STOPWORDS:
                keywords.add(token)

            for i in range(len(token) - 1):
                word = token[i:i + 2]

                if word not in STOPWORDS:
                    keywords.add(word)

        else:
            if len(token) >= 2 and token not in STOPWORDS:
                keywords.add(token)

    return keywords


# ==================== 6. 相关性排序 ====================

def search_knowledge(query, top_k=3):
    """
    本地关键词检索：
    1. 降低常见词的干扰。
    2. 提高区分度较高的关键词权重。
    3. 对完整术语匹配给予加分。
    4. 避免同一页的多个片段占满结果。
    """
    query = query.strip()

    if not query:
        return []

    documents = load_documents()

    if not documents:
        return []

    query_keywords = extract_keywords(query)

    if not query_keywords:
        return []

    # 为每个文档预先计算关键词
    document_keywords = [
        extract_keywords(document["content"])
        for document in documents
    ]

    # 统计每个关键词出现于多少个文档
    document_frequency = Counter()

    for keywords in document_keywords:
        document_frequency.update(keywords)

    total_documents = len(documents)
    results = []

    for document, content_keywords in zip(
        documents, document_keywords
    ):
        matched_keywords = query_keywords & content_keywords

        if not matched_keywords:
            continue

        # 较少见的关键词权重更高
        weighted_total = sum(
            math.log(
                1 + total_documents /
                (1 + document_frequency.get(keyword, 0))
            )
            for keyword in query_keywords
        )

        weighted_matched = sum(
            math.log(
                1 + total_documents /
                (1 + document_frequency.get(keyword, 0))
            )
            for keyword in matched_keywords
        )

        score = (
            weighted_matched / weighted_total
            if weighted_total > 0 else 0
        )

        content_lower = document["content"].lower()
        query_lower = query.lower()

        # 完整查询出现时适当加分
        if len(query_lower) >= 3 and query_lower in content_lower:
            score += 0.5

        # 关键英文术语完整匹配时给予额外加分
        query_terms = re.findall(
            r"[a-zA-Z][a-zA-Z0-9-]*",
            query_lower
        )

        for term in query_terms:
            if len(term) >= 2 and term in content_lower:
                score += 0.15

        results.append({
            **document,
            "score": round(score, 4),
            "matched_keywords": sorted(matched_keywords),
        })

    # 初步排序
    results.sort(
        key=lambda item: (
            item["score"],
            len(item["matched_keywords"])
        ),
        reverse=True
    )

    # 结果多样化：避免同一页反复出现
    selected = []
    selected_keys = set()
    source_counts = Counter()

    # 第一轮：优先不同页面和不同来源
    for item in results:
        source = item["source"]
        page = item.get("page")

        if page is not None:
            unique_key = (source, page)
        else:
            unique_key = (source, item["chunk_id"])

        if unique_key in selected_keys:
            continue

        # 避免同一篇文献占据全部名额
        if source_counts[source] >= 2:
            continue

        selected.append(item)
        selected_keys.add(unique_key)
        source_counts[source] += 1

        if len(selected) >= max(1, top_k):
            break

    # 如果不同来源的候选不足，再补充剩余片段
    if len(selected) < max(1, top_k):
        for item in results:
            if item in selected:
                continue

            selected.append(item)

            if len(selected) >= max(1, top_k):
                break

    return selected


# ==================== 7. 命令行测试 ====================

if __name__ == "__main__":
    print("===== 半导体失效分析：本地 RAG 检索测试 =====")
    print("模式：TXT + PDF，本地关键词检索，不调用大模型 API。")

    documents = load_documents()

    txt_count = sum(
        1 for item in documents
        if item.get("document_type") == "txt"
    )
    pdf_count = sum(
        1 for item in documents
        if item.get("document_type") == "pdf"
    )

    print(f"TXT 知识片段：{txt_count}")
    print(f"PDF 知识片段：{pdf_count}")
    print(f"总计：{len(documents)}")

    while True:
        query = input("\n请输入问题（输入 q 退出）：").strip()

        if query.lower() == "q":
            print("检索结束。")
            break

        if not query:
            print("请输入一个问题。")
            continue

        results = search_knowledge(query, top_k=3)

        if not results:
            print("没有找到匹配的知识片段。")
            continue

        print(f"\n找到 {len(results)} 条候选知识：")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- 候选 {rank} ---")
            print(f"来源：{result['source']}")

            if result.get("page") is not None:
                print(f"页码：{result['page']}")

            print(f"文档类型：{result['document_type']}")
            print(f"片段编号：{result['chunk_id']}")
            print(f"匹配分数：{result['score']}")
            print(
                "匹配关键词："
                + ", ".join(result["matched_keywords"])
            )
            print("内容：")
            print(result["content"])

def _bm25_tokenize(text):
    """
    中文使用 jieba 分词，英文术语按完整单词处理。
    例如：高温高湿、温度循环、FIB-SEM、EDS。
    """
    text = str(text).lower()

    tokens = re.findall(
        r"[\u4e00-\u9fff]+|[a-zA-Z][a-zA-Z0-9-]*|\d+(?:\.\d+)?",
        text
    )

    result = []

    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            # 中文词组进一步分词
            words = jieba.lcut(token)
            result.extend(
                word.strip()
                for word in words
                if word.strip() and len(word.strip()) > 1
            )
        else:
            result.append(token)

    return result


def search_knowledge_bm25(query, top_k=3):
    """
    BM25 检索版本。
    保留原有文档字段，并尽量维持原有结果多样化策略。
    """
    query = query.strip()

    if not query:
        return []

    documents = load_documents()

    if not documents:
        return []

    # 对查询和所有知识库片段进行相同规则的分词
    tokenized_query = _bm25_tokenize(query)
    tokenized_documents = [
        _bm25_tokenize(document.get("content", ""))
        for document in documents
    ]

    if not tokenized_query or not any(tokenized_documents):
        return []

    # 建立 BM25 评分模型
    bm25 = BM25Okapi(tokenized_documents)

    # 计算每个文档与查询的相关性分数
    scores = bm25.get_scores(tokenized_query)

    results = []

    for document, tokens, score in zip(
        documents, tokenized_documents, scores
    ):
        # 没有实际匹配或得分不大于 0 的片段暂不返回
        matched_keywords = set(tokenized_query) & set(tokens)

        if not matched_keywords or score <= 0:
            continue

        results.append({
            **document,
            "score": round(float(score), 4),
            "matched_keywords": sorted(matched_keywords),
        })

    # 按 BM25 得分排序
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # 沿用原有策略，避免同一来源或页面占满结果
    selected = []
    selected_keys = set()
    source_counts = Counter()
    limit = max(1, top_k)

    for item in results:
        source = item["source"]
        page = item.get("page")

        if page is not None:
            unique_key = (source, page)
        else:
            unique_key = (source, item.get("chunk_id"))

        if unique_key in selected_keys:
            continue

        if source_counts[source] >= 2:
            continue

        selected.append(item)
        selected_keys.add(unique_key)
        source_counts[source] += 1

        if len(selected) >= limit:
            break

    # 候选不足时补充其他结果
    if len(selected) < limit:
        for item in results:
            if item in selected:
                continue

            selected.append(item)

            if len(selected) >= limit:
                break

    return selected
