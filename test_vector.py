
from vector_retriever import search_knowledge_vector


QUESTIONS = [
    "器件经过湿热老化后为什么会出现发光性能下降？",
    "FIB-SEM 在先进封装失效分析中有什么用途？",
    "为什么不能仅凭检测到某种元素就认定失效机理？",
]


def main():
    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("问题：", question)
        print("=" * 70)

        results = search_knowledge_vector(question, top_k=3)

        if not results:
            print("没有检索到结果。")
            continue

        for rank, item in enumerate(results, start=1):
            print(f"\n第 {rank} 条")
            print("来源：", item.get("source", "未知"))
            print("页码：", item.get("page", "未知"))
            print("相似度：", item.get("score"))
            print("内容：", item.get("content", "")[:300])


if __name__ == "__main__":
    main()
