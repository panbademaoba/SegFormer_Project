from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(r"/")

AUDIT_DIR = PROJECT_ROOT / "work_dirs" / "R0_audit"

# 用于确定最终 OOD 的审核结果
REVIEW_CSV = AUDIT_DIR / "region_split_review.csv"

# 用于获取全部有效地区
DATASET_CSV = AUDIT_DIR / "dataset_report.csv"

# 最终固定的区域划分文件
OUTPUT_JSON = AUDIT_DIR / "split_region.json"


# ============================================================
# 2. 固定最终 OOD 地区
#
#    来源：
#    region_split_review.csv
#    前16个类别完整的地区
# ============================================================

OOD_REGIONS = [
    "zanzibar",
    "daressalaam",
    "coxsbazar",
    "monrovia",
    "ngaoundere",
    "koeln",
    "kyoto",
    "mahe",
    "melbourne",
    "kagera",
    "rio",
    "shanghai",
    "svaneti",
    "tokyo",
    "tonga",
    "tyrolw",
]


# ============================================================
# 3. 检查输入文件
# ============================================================

if not REVIEW_CSV.exists():
    raise FileNotFoundError(
        f"找不到文件：{REVIEW_CSV}"
    )

if not DATASET_CSV.exists():
    raise FileNotFoundError(
        f"找不到文件：{DATASET_CSV}"
    )


# ============================================================
# 4. 读取审核结果
#
#    用于检查 OOD 地区是否存在
# ============================================================

review_df = pd.read_csv(REVIEW_CSV)

if "region" not in review_df.columns:
    raise ValueError(
        "region_split_review.csv 中没有找到 'region' 列。"
    )


# ============================================================
# 5. 读取全部地区
#
#    注意：
#    这里不能使用 region_split_review.csv，
#    因为它只有19个 OOD 候选地区。
#
#    必须从 dataset_report.csv 获取全部97个地区。
# ============================================================

dataset_df = pd.read_csv(DATASET_CSV)

if "region" not in dataset_df.columns:
    raise ValueError(
        "dataset_report.csv 中没有找到 'region' 列。"
    )


# 获取全部地区
all_regions = sorted(
    dataset_df["region"]
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


# ============================================================
# 6. 检查 OOD 地区
# ============================================================

missing_ood = [
    region
    for region in OOD_REGIONS
    if region not in all_regions
]

if missing_ood:
    raise ValueError(
        "以下 OOD 地区没有在 dataset_report.csv 中找到：\n"
        + "\n".join(missing_ood)
    )


# ============================================================
# 7. 检查 OOD 地区是否确实来自审核候选
# ============================================================

review_regions = set(
    review_df["region"]
    .astype(str)
    .str.strip()
    .tolist()
)

not_in_review = [
    region
    for region in OOD_REGIONS
    if region not in review_regions
]

if not_in_review:
    raise ValueError(
        "以下 OOD 地区没有出现在 region_split_review.csv 中：\n"
        + "\n".join(not_in_review)
    )


# ============================================================
# 8. 剩余地区作为训练候选区域
# ============================================================

train_candidates = [
    region
    for region in all_regions
    if region not in OOD_REGIONS
]


# ============================================================
# 9. 基本数量检查
# ============================================================

print("=" * 60)
print("固定地理区域划分")
print("=" * 60)

print(f"全部地区数量       : {len(all_regions)}")
print(f"OOD 地区数量       : {len(OOD_REGIONS)}")
print(f"训练候选地区数量   : {len(train_candidates)}")

ood_ratio = len(OOD_REGIONS) / len(all_regions)

print(f"OOD 地区比例       : {ood_ratio:.2%}")


# ============================================================
# 10. 输出 OOD 地区
# ============================================================

print("\nOOD 地区：")

for i, region in enumerate(OOD_REGIONS, start=1):
    print(f"{i:2d}. {region}")


# ============================================================
# 11. 输出训练候选地区
# ============================================================

print("\n训练候选地区：")

for i, region in enumerate(train_candidates, start=1):
    print(f"{i:2d}. {region}")


# ============================================================
# 12. 检查数量
# ============================================================

if len(all_regions) != 97:
    print(
        f"\n注意：当前 dataset_report.csv 中读取到 "
        f"{len(all_regions)} 个地区，而预期为97个。"
    )

if len(OOD_REGIONS) != 16:
    raise ValueError(
        f"OOD 地区数量异常：{len(OOD_REGIONS)}，预期为16。"
    )

if len(train_candidates) != len(all_regions) - len(OOD_REGIONS):
    raise ValueError(
        "OOD 地区和训练候选地区的数量关系异常。"
    )

if not (0.15 <= ood_ratio <= 0.20):
    raise ValueError(
        f"OOD 地区比例 {ood_ratio:.2%} 不在15%~20%范围内。"
    )


# ============================================================
# 13. 构建 split_region.json
#
#    当前阶段只固定：
#    - OOD 地区
#    - train_candidates
#
#    IID 暂时不写入。
# ============================================================

split_region = {
    "ood": OOD_REGIONS,
    "train_candidates": train_candidates
}


# ============================================================
# 14. 保存 JSON
# ============================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        split_region,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 15. 重新读取并检查
# ============================================================

with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
    check_data = json.load(f)


print("\n" + "=" * 60)
print("保存完成")
print("=" * 60)

print(f"输出文件：{OUTPUT_JSON}")

print(
    f"\nJSON 中 OOD 地区数量："
    f"{len(check_data['ood'])}"
)

print(
    f"JSON 中训练候选地区数量："
    f"{len(check_data['train_candidates'])}"
)

print("\nR0 地理区域固定完成。")
print("下一步：从 train_candidates 内部固定 IID Validation。")