
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from tools import sem_analysis_tool, rag_search_tool


load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not API_KEY:
    raise ValueError(
        "没有读取到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件。"
    )

# 创建客户端不会发送 API 请求
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# 每轮 Agent 运行的工具调用限制
MAX_SEM_CALLS = 1
MAX_RAG_CALLS = 2


SEM_TOOL = {
    "type": "function",
    "function": {
        "name": "sem_analysis",
        "description": (
            "分析用户提供的 SEM 图像，提取图像尺寸、灰度统计、"
            "边缘比例等基础特征。不能直接确认具体失效类型。"
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
}


RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "检索本地半导体失效分析知识库。适用于 SEM 裂纹、"
            "孔洞、EDS、可靠性测试和失效机理等问题。"
            "返回知识片段及其来源，不直接证明当前样品存在某种缺陷。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要检索的具体问题或关键词"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回知识片段数量，范围为 1 到 5"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}


def validate_analysis(tool_results, ai_message, tool_was_requested=False):
    """检查最终回答及工具执行情况。"""

    issues = []

    if not isinstance(ai_message, str) or not ai_message.strip():
        issues.append("AI 最终回答为空。")

    if tool_was_requested and not tool_results:
        issues.append("模型请求了工具，但没有产生工具执行记录。")

    for item in tool_results:
        result = item.get("result", {})
        tool_name = item.get("tool", "unknown")

        if not isinstance(result, dict):
            issues.append(f"工具 {tool_name} 返回的数据格式异常。")
        elif result.get("status") != "success":
            issues.append(
                f"工具 {tool_name} 执行失败："
                f"{result.get('message', '未提供错误信息')}"
            )

    return {
        "passed": len(issues) == 0,
        "issues": issues
    }


def normalize_query(query):
    """统一空白字符，识别简单的重复检索问题。"""

    return re.sub(r"\s+", " ", query).strip().casefold()


def execute_tool(tool_name, arguments, image_bytes=None):
    """执行本地工具并校验参数。"""

    if tool_name == "sem_analysis":
        if image_bytes is None:
            return {
                "status": "error",
                "message": "SEM 分析需要先提供图像。"
            }

        return sem_analysis_tool(image_bytes)

    if tool_name == "rag_search":
        query = arguments.get("query", "")

        if not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "message": "知识检索问题不能为空。"
            }

        top_k = arguments.get("top_k", 3)

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 5
        ):
            return {
                "status": "error",
                "message": "top_k 必须是 1 到 5 之间的整数。"
            }

        return rag_search_tool(query.strip(), top_k=top_k)

    return {
        "status": "error",
        "message": f"不支持的工具：{tool_name}"
    }


class SupervisorAgent:

    def run(self, user_input, image_bytes=None):
        """执行一轮 Agent 分析。"""

        system_prompt = """
你是一名辅助半导体失效分析的 AI 助手。

可用工具：
1. sem_analysis：提取用户 SEM 图像的基础特征。
2. rag_search：检索本地失效分析知识库。

工具调用规则：

- 当用户提供了 SEM 图像时，必须首先调用 sem_analysis 工具，获取该图像的实际基础特征。
- 当用户的问题涉及 SEM 形貌、裂纹判断、可靠性测试或失效机理时，应调用 rag_search 检索相关知识。
- 当用户同时提供图像并询问失效分析时，应先分析图像，再检索相关知识，最后综合两类工具的实际返回结果作答。
- 如果用户没有提供图像，不得声称已经分析图像。
- 每轮最多调用一次 sem_analysis，最多执行两次不同问题的 RAG 检索。
- 不要重复检索相同问题；若已有足够证据，不要继续检索。
- 不得声称未执行的工具已经执行。

证据规则：
- 区分图像客观特征、候选失效机制和已确认结论。
- 灰度统计和边缘比例只能描述图像特征，不能单独证明裂纹、
  孔洞、分层或其他具体缺陷。
- 知识库内容是辅助资料，不代表当前样品的实验事实。
- 候选机制必须说明依据、局限及需要进一步验证的内容。
- 不得编造数据、实验结果、文献、来源或置信度。
- 引用知识库时，尽可能使用工具实际返回的文件名和片段编号。
- 若知识库未检索到相关内容，明确说明本地资料不足。
- 不能将尚未开展的验证测试描述为已经完成。

当用户要求 SEM 图像或失效分析时，按以下结构回答：
一、客观观测
二、候选失效机制
三、证据与局限
四、当前结论
五、建议验证方案
六、知识来源

如果没有要求图像或失效分析，则根据问题正常回答。
不要强行套用上述报告结构。
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"当前是否已提供 SEM 图像："
                    f"{'是' if image_bytes is not None else '否'}\n"
                    f"用户请求：{user_input}"
                )
            }
        ]

        available_tools = [RAG_TOOL]

        if image_bytes is not None:
            available_tools.append(SEM_TOOL)

        try:
            # 第一次模型请求：判断是否需要工具
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                stream=False
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            # 不需要工具：直接返回模型回答
            if not tool_calls:
                ai_message = message.content or ""

                validation = validate_analysis(
                    tool_results=[],
                    ai_message=ai_message
                )

                return {
                    "status": (
                        "success" if validation["passed"] else "error"
                    ),
                    "selected_tool": [],
                    "tool_results": [],
                    "message": (
                        ai_message.strip()
                        if ai_message.strip()
                        else "模型没有返回文本。"
                    ),
                    "validation": validation
                }

            # 保存模型原始工具调用消息，供后续工具结果对应使用
            messages.append(message.model_dump(exclude_none=True))

            tool_results = []
            rag_cache = {}
            sem_calls = 0
            rag_calls = 0

            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是 JSON 对象。")

                    # SEM 工具最多执行一次
                    if tool_name == "sem_analysis":
                        if sem_calls >= MAX_SEM_CALLS:
                            tool_result = {
                                "status": "error",
                                "message": "本轮 SEM 分析调用次数已达上限。"
                            }
                        else:
                            sem_calls += 1
                            tool_result = execute_tool(
                                tool_name,
                                arguments,
                                image_bytes=image_bytes
                            )

                    # RAG 去重，并限制不同检索问题的数量
                    elif tool_name == "rag_search":
                        query = arguments.get("query", "")

                        if not isinstance(query, str) or not query.strip():
                            tool_result = {
                                "status": "error",
                                "message": "知识检索问题不能为空。"
                            }
                        else:
                            query_key = normalize_query(query)

                            if query_key in rag_cache:
                                # 相同问题复用结果，不再次检索
                                tool_result = dict(rag_cache[query_key])
                                tool_result["cached"] = True

                            elif rag_calls >= MAX_RAG_CALLS:
                                tool_result = {
                                    "status": "error",
                                    "message": (
                                        "本轮不同问题的 RAG 检索次数已达上限。"
                                    )
                                }

                            else:
                                rag_calls += 1
                                tool_result = execute_tool(
                                    tool_name,
                                    arguments,
                                    image_bytes=image_bytes
                                )

                                rag_cache[query_key] = dict(tool_result)

                    else:
                        tool_result = {
                            "status": "error",
                            "message": f"不支持的工具：{tool_name}"
                        }

                except Exception as e:
                    tool_result = {
                        "status": "error",
                        "message": f"{type(e).__name__}: {e}"
                    }

                tool_results.append({
                    "tool": tool_name,
                    "result": tool_result
                })

                # 每个工具调用都必须有对应的 tool 消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str
                    )
                })

            # 第二次模型请求：根据工具结果生成总结
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=False
            )

            ai_message = (
                final_response.choices[0].message.content or ""
            )

            validation = validate_analysis(
                tool_results=tool_results,
                ai_message=ai_message,
                tool_was_requested=True
            )

            tools_succeeded = all(
                isinstance(item.get("result"), dict)
                and item["result"].get("status") == "success"
                for item in tool_results
            )

            if not validation["passed"]:
                status = "error"
            elif tools_succeeded:
                status = "success"
            else:
                status = "partial"

            return {
                "status": status,
                "selected_tool": [
                    item["tool"] for item in tool_results
                ],
                "tool_results": tool_results,
                "message": (
                    ai_message.strip()
                    if ai_message.strip()
                    else "工具已执行，但模型没有返回总结。"
                ),
                "validation": validation
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"{type(e).__name__}: {e}",
                "validation": {
                    "passed": False,
                    "issues": ["Agent 执行过程中发生异常。"]
                }
            }
