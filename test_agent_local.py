from agent import validate_analysis


def run_test(name, tool_results, ai_message, tool_was_requested):
    """只测试本地校验逻辑，不调用大模型 API。"""
    print(f"\n========== {name} ==========")

    try:
        result = validate_analysis(
            tool_results=tool_results,
            ai_message=ai_message,
            tool_was_requested=tool_was_requested,
        )
        print("校验结果：", result)
    except Exception as e:
        print("测试出现异常：", type(e).__name__, str(e))


# 测试 1：普通问答，不需要调用工具
run_test(
    name="测试 1：普通问答",
    tool_results=[],
    ai_message="SEM 中的裂纹需要结合形貌和其他证据综合判断。",
    tool_was_requested=False,
)

# 测试 2：工具执行成功，并返回分析结果
run_test(
    name="测试 2：工具执行成功",
    tool_results=[
        {
            "tool": "sem_analysis",
            "result": {
                "status": "success",
                "image_info": {"width": 526, "height": 395},
                "observations": ["图像中存在边缘结构"],
            },
        }
    ],
    ai_message="已获得图像基础特征，但目前不能据此确认具体缺陷类型。",
    tool_was_requested=True,
)

# 测试 3：要求调用工具，但没有任何工具结果
run_test(
    name="测试 3：工具结果缺失",
    tool_results=[],
    ai_message="分析完成。",
    tool_was_requested=True,
)

# 测试 4：工具执行失败
run_test(
    name="测试 4：工具执行失败",
    tool_results=[
        {
            "tool": "sem_analysis",
            "result": {
                "status": "error",
                "message": "模拟的本地工具错误",
            },
        }
    ],
    ai_message="图像分析未能完成。",
    tool_was_requested=True,
)

print("\n本地模拟测试结束：以上测试均未主动调用 DeepSeek API。")