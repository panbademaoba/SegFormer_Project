import pandas as pd
import os


# =========================
# 路径
# =========================

AUDIT_DIR = r"D:\SegFormer_Project\work_dirs\R0_audit"


pair_file = os.path.join(
    AUDIT_DIR,
    "region_pair_check.csv"
)

size_file = os.path.join(
    AUDIT_DIR,
    "region_size_check.csv"
)


save_file = os.path.join(
    AUDIT_DIR,
    "dataset_report.csv"
)


# =========================
# 读取数据
# =========================

pair_df = pd.read_csv(pair_file)

size_df = pd.read_csv(size_file)



# =========================
# 有效pair统计
# =========================

valid_pairs = (
    size_df
    .groupby("region")
    .size()
    .reset_index(name="valid_pairs")
)



# =========================
# 合并
# =========================

report = pd.merge(
    pair_df,
    valid_pairs,
    on="region",
    how="left"
)


report["valid_pairs"] = (
    report["valid_pairs"]
    .fillna(0)
    .astype(int)
)



# =========================
# 保存
# =========================

report.to_csv(
    save_file,
    index=False,
    encoding="utf-8-sig"
)



print("dataset_report.csv生成完成")

print(save_file)

print(report.head())