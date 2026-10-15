
import csv
from pathlib import Path

from vector_retriever import search_knowledge_vector


# 评测问题：覆盖当前项目中的典型半导体失效分析场景
TEST_QUESTIONS = [
    "湿热老化后发光性能下降，可能与哪些因素有关？",
    "FIB-SEM 在先进封装失效分析中有什么用途？",
    "为什么不能仅凭 EDS 检测到某种元素就认定失效机理？",
    "SEM 可以观察 Micro-LED 图案化器件的哪些结构特征？",
    "如何比较器件老化前后的表面形貌变化？",
    "85°C/85%RH 高温高湿试验主要评价什么？",
    "温度循环试验可能导致哪些器件可靠性问题？",
    "如何判断发光强度下降是否与材料结构变化有关？",
    "XRD 和 SEM 在材料分析中的作用有什么不同？",
    "失效分析中如何结合多种表征结果验证候选机理？",
]


def main():
    output_file = Path(__file__).parent / "retrieval_eval_results.csv"

    rows = []

    for index, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{index}/{len(TEST_QUESTIONS)}] 正在检索：{question}")

        try:
            results = search_knowledge_vector(question, top_k=5)

            if not results:
                print("  没有检索到结果")
                rows.append({
                    "question_id": index,
                    "question": question,
                    "rank": "",
                    "source": "",
                    "page": "",
                    "score": "",
                    "content": "",
                    "relevant": "",
                })
                continue

            for rank, item in enumerate(results, start=1):
                row = {
                    "question_id": index,
                    "question": question,
                    "rank": rank,
                    "source": item.get("source", ""),
                    "page": item.get("page", ""),
                    "score": item.get("score", ""),
                    "content": item.get("content", ""),
                    # 后续由你人工填写：1=相关，0=不相关
                    "relevant": "",
                }
                rows.append(row)

                print(
                    f"  Top-{rank} | "
                    f"分数={row['score']} | "
                    f"来源={row['source']} | "
                    f"页码={row['page']}"
                )

        except Exception as exc:
            print(f"  检索失败：{exc}")
            print("  请先检查 vector_retriever.py 是否能正常运行。")
            raise

    fieldnames = [
        "question_id",
        "question",
        "rank",
        "source",
        "page",
        "score",
        "content",
        "relevant",
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n评测结果已保存：")
    print(output_file)
    print(f"测试问题数：{len(TEST_QUESTIONS)}")
    print(f"检索记录数：{len(rows)}")
    print("\n下一步：打开 CSV，阅读每条 content，人工填写 relevant 列。")


if __name__ == "__main__":
    main()
