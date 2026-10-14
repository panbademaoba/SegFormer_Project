from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(r"/")

AUDIT_DIR = PROJECT_ROOT / "work_dirs" / "R0_audit"

SPLIT_JSON = AUDIT_DIR / "split_region.json"
DATASET_CSV = AUDIT_DIR / "dataset_report.csv"

OUTPUT_CSV = AUDIT_DIR / "iid_pool_statistics.csv"


# ============================================================
# 2. 检查输入文件
# ============================================================

if not SPLIT_JSON.exists():
    raise FileNotFoundError(
        f"找不到文件：{SPLIT_JSON}"
    )

if not DATASET_CSV.exists():
    raise FileNotFoundError(
        f"找不到文件：{DATASET_CSV}"
    )


# ============================================================
# 3. 读取 split_region.json
# ============================================================

with open(SPLIT_JSON, "r", encoding="utf-8") as f:
    split_data = json.load(f)


if "train_candidates" not in split_data:
    raise ValueError(
        "split_region.json 中没有找到 train_candidates。"
    )


train_candidates = [
    str(region).strip()
    for region in split_data["train_candidates"]
]


# ============================================================
# 4. 读取 dataset_report.csv
# ============================================================

df = pd.read_csv(DATASET_CSV)

required_columns = [
    "region",
    "valid_pairs"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"dataset_report.csv 中缺少字段：{column}"
        )


# ============================================================
# 5. 清理地区名称
# ============================================================

df["region"] = (
    df["region"]
    .astype(str)
    .str.strip()
)

df["valid_pairs"] = pd.to_numeric(
    df["valid_pairs"],
    errors="coerce"
).fillna(0)


# ============================================================
# 6. 只保留 train_candidates
# ============================================================

train_df = df[
    df["region"].isin(train_candidates)
].copy()


# ============================================================
# 7. 每个地区统计
# ============================================================

region_stats = (
    train_df[
        ["region", "valid_pairs"]
    ]
    .drop_duplicates(subset=["region"])
    .copy()
)


# ============================================================
# 8. 检查是否存在重复地区
# ============================================================

if train_df["region"].duplicated().any():
    print("警告：dataset_report.csv 中存在重复地区记录。")
    print("将对重复地区的 valid_pairs 进行求和。")

    region_stats = (
        train_df
        .groupby("region", as_index=False)["valid_pairs"]
        .sum()
    )


# ============================================================
# 9. 补齐 split_region.json 中可能没有出现在 CSV 的地区
# ============================================================

existing_regions = set(
    region_stats["region"].tolist()
)

missing_regions = [
    region
    for region in train_candidates
    if region not in existing_regions
]

if missing_regions:
    print("\n警告：以下 train_candidates 没有出现在 dataset_report.csv：")
    for region in missing_regions:
        print(f"  - {region}")

    missing_df = pd.DataFrame({
        "region": missing_regions,
        "valid_pairs": 0
    })

    region_stats = pd.concat(
        [region_stats, missing_df],
        ignore_index=True
    )


# ============================================================
# 10. 重新排序
# ============================================================

region_stats = region_stats.sort_values(
    by="valid_pairs",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 11. 增加状态字段
# ============================================================

region_stats["usable_for_train"] = (
    region_stats["valid_pairs"] > 0
)


# ============================================================
# 12. 统计总体情况
# ============================================================

total_train_regions = len(train_candidates)

usable_regions = int(
    (region_stats["valid_pairs"] > 0).sum()
)

zero_regions = int(
    (region_stats["valid_pairs"] == 0).sum()
)

total_valid_pairs = int(
    region_stats["valid_pairs"].sum()
)


# ============================================================
# 13. 输出统计信息
# ============================================================

print("=" * 65)
print("IID / Train Pool 候选数据统计")
print("=" * 65)

print(f"train_candidates 地区数量 : {total_train_regions}")
print(f"有有效影像的地区数量      : {usable_regions}")
print(f"valid_pairs = 0 的地区数   : {zero_regions}")
print(f"有效 image-label 对总数    : {total_valid_pairs}")

print("\n" + "-" * 65)
print("各地区有效 image-label 对数量")
print("-" * 65)

for _, row in region_stats.iterrows():
    region = row["region"]
    valid_pairs = int(row["valid_pairs"])

    status = "可用" if valid_pairs > 0 else "不可用"

    print(
        f"{region:<25} "
        f"{valid_pairs:>5}    "
        f"{status}"
    )


# ============================================================
# 14. 保存 CSV
# ============================================================

region_stats.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 最终提示
# ============================================================

print("\n" + "=" * 65)
print("分析完成")
print("=" * 65)

print(f"输出文件：{OUTPUT_CSV}")

print("\n注意：")
print("本脚本只进行 IID / Train Pool 候选数据统计。")
print("没有随机抽取 IID，也没有修改 split_region.json。")
print("下一步根据本统计结果再确定 IID Validation 的具体方案。")