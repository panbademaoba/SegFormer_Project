from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(r"/")

AUDIT_DIR = PROJECT_ROOT / "work_dirs" / "R0_audit"

SPLIT_JSON = AUDIT_DIR / "split_region.json"

METADATA_CSV = AUDIT_DIR / "metadata.csv"


TRAIN_POOL_TXT = AUDIT_DIR / "train_pool.txt"

IID_VAL_TXT = AUDIT_DIR / "iid_val.txt"

STATISTICS_CSV = AUDIT_DIR / "iid_split_statistics.csv"


# ============================================================
# 2. 参数
# ============================================================

IID_RATIO = 0.2

RANDOM_SEED = 42


# ============================================================
# 3. 检查文件
# ============================================================

for file in [
    SPLIT_JSON,
    METADATA_CSV
]:
    if not file.exists():
        raise FileNotFoundError(
            f"找不到文件: {file}"
        )


# ============================================================
# 4. 读取区域划分
# ============================================================

with open(
    SPLIT_JSON,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


ood_regions = set(
    split_data["ood"]
)


train_candidates = set(
    split_data["train_candidates"]
)


# ============================================================
# 5. 读取 metadata
# ============================================================

df = pd.read_csv(
    METADATA_CSV
)


required_cols = [
    "region",
    "image_path",
    "label_path"
]


for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"metadata.csv 缺少字段: {col}"
        )


# ============================================================
# 6. 去除 OOD
# ============================================================

train_candidate_df = df[
    df["region"].isin(train_candidates)
].copy()


# 再保险检查
train_candidate_df = train_candidate_df[
    ~train_candidate_df["region"].isin(ood_regions)
]


print("=" * 60)
print("IID 划分开始")
print("=" * 60)

print(
    f"全部 metadata 样本数: {len(df)}"
)

print(
    f"OOD 样本数量: "
    f"{len(df[df['region'].isin(ood_regions)])}"
)

print(
    f"训练候选样本数量: "
    f"{len(train_candidate_df)}"
)


# ============================================================
# 7. 固定随机划分
# ============================================================

train_pool, iid_val = train_test_split(
    train_candidate_df,
    test_size=IID_RATIO,
    random_state=RANDOM_SEED,
    shuffle=True
)


# ============================================================
# 8. 保存 txt
# ============================================================

with open(
    TRAIN_POOL_TXT,
    "w",
    encoding="utf-8"
) as f:

    for path in train_pool["image_path"]:
        f.write(
            str(path)
            + "\n"
        )


with open(
    IID_VAL_TXT,
    "w",
    encoding="utf-8"
) as f:

    for path in iid_val["image_path"]:
        f.write(
            str(path)
            + "\n"
        )


# ============================================================
# 9. 统计地区分布
# ============================================================

train_stat = (
    train_pool
    .groupby("region")
    .size()
    .reset_index(
        name="train_count"
    )
)


val_stat = (
    iid_val
    .groupby("region")
    .size()
    .reset_index(
        name="iid_count"
    )
)


stats = pd.merge(
    train_stat,
    val_stat,
    on="region",
    how="outer"
)


stats = stats.fillna(0)


stats["train_count"] = (
    stats["train_count"]
    .astype(int)
)


stats["iid_count"] = (
    stats["iid_count"]
    .astype(int)
)


stats = stats.sort_values(
    by="region"
)


stats.to_csv(
    STATISTICS_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 输出结果
# ============================================================

print("\n" + "=" * 60)
print("IID 划分完成")
print("=" * 60)


print(
    f"Train Pool 数量 : {len(train_pool)}"
)

print(
    f"IID Validation 数量 : {len(iid_val)}"
)


print(
    f"\nTrain Pool 文件:"
    f"\n{TRAIN_POOL_TXT}"
)


print(
    f"\nIID Validation 文件:"
    f"\n{IID_VAL_TXT}"
)


print(
    f"\n统计文件:"
    f"\n{STATISTICS_CSV}"
)


print("\n固定参数:")
print(
    f"IID比例={IID_RATIO}"
)

print(
    f"Random Seed={RANDOM_SEED}"
)