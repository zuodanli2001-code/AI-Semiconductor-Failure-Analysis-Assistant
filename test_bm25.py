
from rag_retriever import (
    search_knowledge,
    search_knowledge_bm25,
)

QUESTIONS = [
    "SEM 可以观察哪些半导体器件失效形貌？",
    "EDS 能分析什么？它能否直接确定元素的化学价态？",
    "FIB-SEM 在先进封装失效分析中有什么用途？",
    "如何结合 SEM 和 EDS 分析异常区域？",
    "高温高湿试验可以评价什么可靠性问题？",
    "温度循环试验主要用于评价什么？",
    "为什么不能仅凭检测到某种元素就认定失效机理？",
    "SEM/EDS 微区分析有哪些需要注意的限制？",
]


def show_results(title, results):
    print(f"\n{title}")

    if not results:
        print("没有检索到结果")
        return

    for rank, item in enumerate(results, start=1):
        print(f"\n  第 {rank} 条")
        print(f"  来源：{item.get('source', '未知')}")
        print(f"  页码：{item.get('page', '未知')}")
        print(f"  分数：{item.get('score', '未知')}")
        print(f"  匹配词：{item.get('matched_keywords', [])}")
        print(f"  内容：{item.get('content', '')[:220]}")


def main():
    for i, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 70)
        print(f"问题 {i}：{question}")
        print("=" * 70)

        old_results = search_knowledge(question, top_k=3)
        bm25_results = search_knowledge_bm25(question, top_k=3)

        show_results("【旧版关键词检索】", old_results)
        show_results("【BM25 检索】", bm25_results)


if __name__ == "__main__":
    main()
