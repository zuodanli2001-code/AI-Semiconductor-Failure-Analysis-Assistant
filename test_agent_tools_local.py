from agent import execute_tool, validate_analysis


def show_result(name, result):
    print(f"\n===== {name} =====")
    print(result)


# 测试 1：通过 Agent 的工具分发函数调用本地 RAG
rag_result = execute_tool(
    tool_name="rag_search",
    arguments={
        "query": "SEM裂纹如何判断？",
        "top_k": 3
    }
)
show_result("测试 1：RAG 工具分发", rag_result)

# 测试 2：无匹配结果仍然属于成功检索
empty_result = {
    "status": "success",
    "query": "一个知识库中没有的测试主题",
    "count": 0,
    "results": []
}

show_result(
    "测试 2：无匹配内容的校验",
    validate_analysis(
        tool_results=[
            {"tool": "rag_search", "result": empty_result}
        ],
        ai_message="本地知识库没有找到相关资料。",
        tool_was_requested=True
    )
)

# 测试 3：工具参数错误
bad_argument_result = execute_tool(
    tool_name="rag_search",
    arguments={
        "query": "SEM裂纹",
        "top_k": 100
    }
)
show_result("测试 3：非法参数处理", bad_argument_result)

# 测试 4：未知工具
unknown_result = execute_tool(
    tool_name="unknown_tool",
    arguments={}
)
show_result("测试 4：未知工具处理", unknown_result)

print("\n本地测试结束。本脚本没有调用大模型 API。")