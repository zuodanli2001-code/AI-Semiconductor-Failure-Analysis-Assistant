
from rag_retriever import search_knowledge


# 固定测试集：先检查当前 RAG 能否检索到相关资料
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "SEM 可以观察哪些半导体器件失效形貌？",
        "expected": "裂纹、孔洞、分层等表面或截面形貌信息",
    },
    {
        "id": 2,
        "question": "EDS 能分析什么？它能否直接确定元素的化学价态？",
        "expected": "元素组成；不能仅凭 EDS 直接确定化学价态",
    },
    {
        "id": 3,
        "question": "FIB-SEM 在先进封装失效分析中有什么用途？",
        "expected": "局部截面制备、微观结构观察或缺陷定位等相关内容",
    },
    {
        "id": 4,
        "question": "如何结合 SEM 和 EDS 分析异常区域？",
        "expected": "先观察形貌，再分析元素组成，并结合其他证据",
    },
    {
        "id": 5,
        "question": "高温高湿试验可以评价什么可靠性问题？",
        "expected": "湿热环境下的可靠性及可能发生的性能或结构变化",
    },
    {
        "id": 6,
        "question": "温度循环试验主要用于评价什么？",
        "expected": "温度反复变化造成的性能、结构或材料可靠性问题",
    },
    {
        "id": 7,
        "question": "为什么不能仅凭检测到某种元素就认定失效机理？",
        "expected": "元素检出只是证据之一，需要结合形貌、其他表征和可靠性结果",
    },
    {
        "id": 8,
        "question": "SEM/EDS 微区分析有哪些需要注意的限制？",
        "expected": "EDS 的分析能力及限制、测试条件或结果解释方面的相关内容",
    },
]


def main():
    total = len(TEST_QUESTIONS)
    successful_queries = 0

    print("=" * 70)
    print("RAG 检索基线测试")
    print(f"测试问题数量：{total}")
    print("=" * 70)

    for item in TEST_QUESTIONS:
        question_id = item["id"]
        question = item["question"]

        print(f"\n【问题 {question_id}】{question}")
        print(f"预期检索方向：{item['expected']}")
        print("-" * 70)

        try:
            results = search_knowledge(question, top_k=3)

            if not results:
                print("检索结果：没有找到内容")
                continue

            successful_queries += 1

            for rank, result in enumerate(results, start=1):
                print(f"\n  结果 {rank}")
                print(f"  来源：{result.get('source', '未知')}")
                print(f"  页码：{result.get('page', '未知')}")
                print(f"  分数：{result.get('score', '未提供')}")
                print(
                    "  匹配关键词：",
                    result.get("matched_keywords", [])
                )

                content = result.get("content", "")
                print(f"  内容：{content[:350]}")

        except Exception as exc:
            print(f"检索出错：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("测试运行结束")
    print(f"有返回结果的问题：{successful_queries}/{total}")
    print("注意：有结果不代表检索准确，仍需人工检查内容相关性。")
    print("=" * 70)


if __name__ == "__main__":
    main()
