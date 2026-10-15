
import os
from dotenv import load_dotenv
from openai import OpenAI

# 读取项目根目录下的 .env 文件
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

if not api_key:
    raise ValueError(
        "没有读取到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件。"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个专业、谨慎的半导体失效分析助手。"
            },
            {
                "role": "user",
                "content": "请用一句话介绍 SEM 在半导体失效分析中的作用。"
            }
        ],
        stream=False
    )

    print("\nDeepSeek API 连接成功！")
    print("当前模型：", response.model)
    print("模型回复：")
    print(response.choices[0].message.content)

except Exception as e:
    print("\nAPI 调用失败，请检查错误信息：")
    print(type(e).__name__, str(e))