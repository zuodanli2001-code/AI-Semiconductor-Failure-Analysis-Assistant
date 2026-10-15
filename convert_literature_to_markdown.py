
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 读取项目配置
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "literature" / "processed"
OUTPUT_DIR = BASE_DIR / "literature" / "markdown"
CACHE_FILE = OUTPUT_DIR / "translation_cache.json"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not API_KEY:
    raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查项目根目录的 .env 文件。")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print("翻译缓存读取失败，将重新建立缓存。")
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def translate_text(text):
    """翻译一段英文，尽量保留专业术语、数字和原有结构。"""
    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名半导体、材料科学与失效分析领域的专业科技翻译。"
                    "请将用户提供的英文科技文献准确翻译成中文。"
                    "保留专业术语、缩写、公式、数值、单位、材料名称和引用编号。"
                    "不要总结，不要删减，不要添加原文没有的信息。"
                    "保留段落结构；表格内容尽可能按原有结构呈现。"
                    "只输出中文译文，不要输出翻译说明。"
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content.strip()


def safe_filename(name):
    name = Path(name).stem
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name[:150] or "未命名文献"


def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"没有找到 JSON 文献文件：{INPUT_DIR}")
        return

    cache = load_cache()
    print(f"找到 {len(json_files)} 个 JSON 文件。")
    print(f"Markdown 输出目录：{OUTPUT_DIR}")

    for file_index, json_path in enumerate(json_files, start=1):
        print(f"\n[{file_index}/{len(json_files)}] 正在处理：{json_path.name}")

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"文件读取失败，跳过：{exc}")
            continue

        pages = data.get("pages", [])
        if not pages:
            print("没有找到 pages 字段，跳过该文件。")
            continue

        output_path = OUTPUT_DIR / f"{safe_filename(json_path.name)}.md"
        sections = [
            f"# {Path(json_path.name).stem}",
            "",
            f"> 来源文件：{json_path.name}",
            f"> 总页数：{data.get('total_pages', len(pages))}",
            "",
            "> 说明：英文原文与中文翻译分开保存。翻译由 AI 生成，重要术语和数据建议对照原文核验。",
            "",
        ]

        for page_index, page in enumerate(pages, start=1):
            page_number = page.get("page", page_index)
            english = str(page.get("text", "")).strip()

            if not english:
                continue

            # 以文件名、页码和原文共同确定缓存键
            import hashlib
            cache_key = hashlib.sha256(
                f"{json_path.name}|{page_number}|{english}".encode("utf-8")
            ).hexdigest()

            if cache_key in cache:
                chinese = cache[cache_key]
                print(f"  第 {page_number} 页：使用已有翻译")
            else:
                print(f"  第 {page_number} 页：正在翻译……")
                try:
                    chinese = translate_text(english)
                    cache[cache_key] = chinese
                    save_cache(cache)
                    time.sleep(0.3)
                except Exception as exc:
                    print(f"  第 {page_number} 页翻译失败：{exc}")
                    chinese = "【本页翻译失败，请重新运行脚本重试。】"

            sections.extend([
                f"## 第 {page_number} 页",
                "",
                "### English Original",
                "",
                english,
                "",
                "### 中文翻译",
                "",
                chinese,
                "",
                "---",
                "",
            ])

        output_path.write_text("\n".join(sections), encoding="utf-8")
        print(f"已保存：{output_path.name}")

    print("\n全部文件处理完成。")
    print(f"请检查目录：{OUTPUT_DIR}")
    print("如果某页翻译失败，可重新运行脚本；成功的翻译会从缓存中读取。")


if __name__ == "__main__":
    main()
