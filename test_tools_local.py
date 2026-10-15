import json
from tools import rag_search_tool


def main():
    print("===== RAG 工具本地测试 =====")
    print("本测试只检索本地知识库，不调用 DeepSeek。")

    questions = [
        "SEM裂纹如何判断？",
        "85°C/85%RH可靠性测试",
    ]

    for question in questions:
        print(f"\n问题：{question}")

        result = rag_search_tool(question, top_k=3)

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ))


if __name__ == "__main__":
    main()
