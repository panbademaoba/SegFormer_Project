import os
import pandas as pd

# =========================
# 1. 路径设置
# =========================

PROJECT_ROOT = r"D:\SegFormer_Project"

REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "work_dirs",
    "R0_audit",
    "dataset_report.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "work_dirs",
    "R0_audit",
    "region_split_candidates.csv"
)

# =========================
# 2. 读取 R0 数据审计结果
# =========================

print("读取数据审计结果：")
print(REPORT_PATH)

if not os.path.exists(REPORT_PATH):
    raise FileNotFoundError(
        f"找不到 dataset_report.csv：\n{REPORT_PATH}"
    )

df = pd.read_csv(REPORT_PATH)

print("\n数据读取成功。")
print(f"总地区数：{len(df)}")

# =========================
# 3. 检查必要字段
# =========================

required_columns = [
    "region",
    "images",
    "labels",
    "missing_labels",
    "missing_images",
    "valid_pairs"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"dataset_report.csv 缺少必要字段：{col}"
        )

# =========================
# 4. 只保留有有效 image-label pair 的地区
# =========================

valid_regions = df[df["valid_pairs"] > 0].copy()

valid_regions = valid_regions.sort_values(
    by="valid_pairs",
    ascending=False
).reset_index(drop=True)

# =========================
# 5. 统计信息
# =========================

total_regions = len(df)
valid_region_count = len(valid_regions)

ood_min = int(total_regions * 0.15)
ood_max = int(total_regions * 0.20)

print("\n==============================")
print("地区统计")
print("==============================")

print(f"原始地区数量：{total_regions}")
print(f"存在有效 image-label pair 的地区：{valid_region_count}")

print("\n按照指南约 15%–20% 的地区作为 OOD Test：")
print(f"15% ≈ {ood_min} 个地区")
print(f"20% ≈ {ood_max} 个地区")

# =========================
# 6. 增加排名
# =========================

valid_regions.insert(
    0,
    "region_rank",
    range(1, len(valid_regions) + 1)
)

# =========================
# 7. 增加 OOD 数量参考
# =========================

valid_regions["ood_candidate_15_20pct"] = ""

if len(valid_regions) >= ood_max:
    valid_regions.loc[:ood_max - 1, "ood_candidate_15_20pct"] = "候选"
elif len(valid_regions) >= ood_min:
    valid_regions.loc[:ood_min - 1, "ood_candidate_15_20pct"] = "候选"

# =========================
# 8. 保存候选地区表
# =========================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

valid_regions.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

# =========================
# 9. 输出结果
# =========================

print("\n==============================")
print("地区候选表")
print("==============================")

print(
    valid_regions[
        [
            "region_rank",
            "region",
            "images",
            "labels",
            "missing_labels",
            "missing_images",
            "valid_pairs",
            "ood_candidate_15_20pct"
        ]
    ].to_string(index=False)
)

print("\n==============================")
print("完成")
print("==============================")

print(f"\n候选地区表已经保存到：")
print(OUTPUT_PATH)