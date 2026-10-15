import json
from pathlib import Path

import pymupdf


# 项目目录
BASE_DIR = Path(__file__).resolve().parent

# 原始 PDF 论文目录
LITERATURE_DIR = BASE_DIR / "literature"

# 提取结果目录
PROCESSED_DIR = LITERATURE_DIR / "processed"


def extract_pdf(pdf_path: Path) -> dict:
    """逐页提取 PDF 文本，并保留页码和来源信息。"""

    pages = []

    try:
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                # 跳过完全没有提取到文本的页面
                if not text:
                    continue

                pages.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    }
                )

        return {
            "status": "success",
            "source": pdf_path.name,
            "total_pages": len(pages),
            "pages": pages,
        }

    except Exception as exc:
        return {
            "status": "error",
            "source": pdf_path.name,
            "message": str(exc),
            "pages": [],
        }


def main():
    LITERATURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(LITERATURE_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"没有找到 PDF 文件，请把论文放到：{LITERATURE_DIR}")
        return

    success_count = 0

    for pdf_path in pdf_files:
        print(f"\n正在读取：{pdf_path.name}")

        result = extract_pdf(pdf_path)

        if result["status"] != "success":
            print(f"读取失败：{result['message']}")
            continue

        # 生成 JSON 文件名
        output_path = PROCESSED_DIR / f"{pdf_path.stem}.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        page_count = result["total_pages"]
        char_count = sum(len(page["text"]) for page in result["pages"])

        print(f"读取成功：提取到 {page_count} 页有效文本")
        print(f"文本字符数：{char_count}")
        print(f"保存位置：{output_path}")

        if page_count == 0:
            print("提醒：没有提取到可用文本，可能是扫描版 PDF，需要 OCR。")
        else:
            success_count += 1

    print(f"\n处理结束：成功提取 {success_count} 篇 PDF。")


if __name__ == "__main__":
    main()