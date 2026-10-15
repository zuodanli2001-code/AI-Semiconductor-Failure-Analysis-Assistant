
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean


# ==================== 1. 文件路径 ====================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "retrieval_eval_results_labeled.csv"
DETAIL_FILE = BASE_DIR / "retrieval_eval_per_question.csv"


# ==================== 2. 读取人工标注结果 ====================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到标注文件：{INPUT_FILE}\n"
        "请检查 CSV 文件名是否正确。"
    )


def read_csv_file(path):
    """兼容常见的 CSV 文件编码。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 编码无法识别，请检查文件编码。")


rows = read_csv_file(INPUT_FILE)

if not rows:
    raise ValueError("CSV 文件没有数据。")

required_columns = {
    "question_id", "question", "rank", "relevant"
}

missing = required_columns - set(rows[0].keys())

if missing:
    raise ValueError(
        f"CSV 缺少必要字段：{missing}\n"
        f"实际字段：{list(rows[0].keys())}"
    )


# ==================== 3. 检查并整理数据 ====================

groups = defaultdict(list)

for line_number, row in enumerate(rows, start=2):
    qid = row["question_id"].strip()
    question = row["question"].strip()
    label = row["relevant"].strip()

    if not qid:
        raise ValueError(f"第 {line_number} 行缺少 question_id。")

    if label not in ("0", "1"):
        raise ValueError(
            f"第 {line_number} 行 relevant={label!r}，"
            "请检查是否填写了 0 或 1。"
        )

    try:
        rank = int(row["rank"])
    except (ValueError, TypeError):
        raise ValueError(f"第 {line_number} 行的 rank 不是整数。")

    groups[qid].append({
        "question": question,
        "rank": rank,
        "relevant": int(label),
    })


# ==================== 4. 逐个问题计算指标 ====================

results = []

for qid, items in groups.items():
    items.sort(key=lambda x: x["rank"])

    # 本项目评估前 5 条检索结果
    top5 = [item for item in items if item["rank"] <= 5]

    if len(top5) != 5:
        raise ValueError(
            f"问题 {qid} 的 Top-5 结果不是 5 条，"
            f"实际为 {len(top5)} 条，请检查 CSV。"
        )

    relevant_count = sum(item["relevant"] for item in top5)

    # Precision@5：前 5 条中相关资料的比例
    precision_at_5 = relevant_count / 5

    # Hit@5：前 5 条中是否至少有一条相关资料
    hit_at_5 = int(relevant_count > 0)

    # MRR：第一条相关资料排名的倒数
    relevant_ranks = [
        item["rank"] for item in top5
        if item["relevant"] == 1
    ]

    reciprocal_rank = (
        1 / relevant_ranks[0]
        if relevant_ranks else 0
    )

    results.append({
        "question_id": qid,
        "question": items[0]["question"],
        "relevant_count_at_5": relevant_count,
        "Precision@5": precision_at_5,
        "MRR": reciprocal_rank,
        "Hit@5": hit_at_5,
        "first_relevant_rank": (
            relevant_ranks[0] if relevant_ranks else ""
        ),
    })


# ==================== 5. 计算整体平均指标 ====================

precision_mean = mean(r["Precision@5"] for r in results)
mrr_mean = mean(r["MRR"] for r in results)
hit_mean = mean(r["Hit@5"] for r in results)

total_relevant = sum(r["relevant_count_at_5"] for r in results)
total_retrieved = sum(5 for _ in results)

# ==================== 6. 保存逐问题评估报告 ====================

fieldnames = [
    "question_id",
    "question",
    "relevant_count_at_5",
    "Precision@5",
    "MRR",
    "Hit@5",
    "first_relevant_rank",
]

with open(DETAIL_FILE, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for result in results:
        output = result.copy()

        for key in ("Precision@5", "MRR"):
            output[key] = f"{output[key]:.4f}"

        writer.writerow(output)


# ==================== 7. 输出评估结果 ====================

print("=" * 50)
print("RAG 检索效果评估报告")
print("=" * 50)
print(f"测试问题数：{len(results)}")
print(f"检索结果数：{total_retrieved}")
print(f"相关结果数：{total_relevant}")
print(f"Precision@5：{precision_mean:.4f} ({precision_mean:.2%})")
print(f"MRR：        {mrr_mean:.4f}")
print(f"Hit@5：      {hit_mean:.4f} ({hit_mean:.2%})")
print("-" * 50)

for result in results:
    print(
        f"问题 {result['question_id']} | "
        f"相关数={result['relevant_count_at_5']}/5 | "
        f"Precision@5={result['Precision@5']:.2f} | "
        f"MRR={result['MRR']:.2f} | "
        f"Hit@5={result['Hit@5']}"
    )

print("-" * 50)
print(f"逐问题报告已保存到：{DETAIL_FILE}")
print("评估完成！")
