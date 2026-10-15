
from pathlib import Path

from tools import analyze_sem_tool


# 查找现有测试文件中使用的 SEM 图片路径
# 第一次测试时，请把下面路径替换成你自己的真实图片路径。
image_path = Path(r"F:\你的SEM图片实际路径.png")

if not image_path.is_file():
    print(f"找不到图片，请检查路径：{image_path}")
else:
    image_bytes = image_path.read_bytes()
    result = analyze_sem_tool(image_bytes)
    print(result)