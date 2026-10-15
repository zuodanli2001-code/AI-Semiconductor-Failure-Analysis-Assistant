
from rag_retriever import split_text


def main():
    text = (
        "SEM可用于观察样品表面的裂纹、孔洞和分层等异常形貌。"
        "EDS可用于分析异常区域的元素组成。"
        "EDS检测到某种元素，并不意味着已经确定其化学价态或具体化合物。"
        "复杂失效问题需要结合其他表征手段与可靠性测试结果。"
        "高温高湿试验可用于评价器件在湿热环境下的可靠性。"
        "温度循环试验可用于观察温度变化引起的性能与结构变化。"
        "分析失效原因时，应区分实验观察结果、候选机理和已经验证的结论。"
    )

    max_length = 100
    overlap = 30

    chunks = split_text(
        text,
        max_length=max_length,
        overlap=overlap
    )

    print(f"切片数量：{len(chunks)}")
    print(f"最大允许长度：{max_length}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n切片 {i}，长度={len(chunk)}")
        print("开头：", chunk[:45])
        print("结尾：", chunk[-45:])

    # 自动验收
    assert all(len(c) <= max_length for c in chunks), \
        "失败：存在超过最大长度的切片"

    assert all(c.strip() for c in chunks), \
        "失败：存在空切片"

    assert len(chunks) == len(set(chunks)), \
        "失败：存在完全重复的切片"

    assert "".join(chunks).replace(" ", "") != "", \
        "失败：切片内容为空"

    print("\n基础验收通过：长度、空切片、完全重复检查均通过。")


if __name__ == "__main__":
    main()
