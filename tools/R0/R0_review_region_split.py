import os
import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = r"D:\SegFormer_Project"

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "work_dirs",
    "R0_audit"
)

CANDIDATE_PATH = os.path.join(
    AUDIT_DIR,
    "region_split_candidates.csv"
)

COVERAGE_PATH = os.path.join(
    AUDIT_DIR,
    "region_class_coverage.csv"
)

OUTPUT_PATH = os.path.join(
    AUDIT_DIR,
    "region_split_review.csv"
)


# ============================================================
# 2. 读取数据
# ============================================================

print("=" * 60)
print("R0 地区划分方案审查")
print("=" * 60)

if not os.path.exists(CANDIDATE_PATH):
    raise FileNotFoundError(
        f"找不到：\n{CANDIDATE_PATH}"
    )

if not os.path.exists(COVERAGE_PATH):
    raise FileNotFoundError(
        f"找不到：\n{COVERAGE_PATH}"
    )


candidate_df = pd.read_csv(
    CANDIDATE_PATH
)

coverage_df = pd.read_csv(
    COVERAGE_PATH
)


# ============================================================
# 3. 获取当前 OOD 候选
# ============================================================

candidate_df = candidate_df[
    candidate_df["ood_candidate_15_20pct"] == "候选"
].copy()


# ============================================================
# 4. 合并类别覆盖信息
# ============================================================

review_df = candidate_df[
    [
        "region",
        "images",
        "labels",
        "missing_labels",
        "valid_pairs"
    ]
].merge(
    coverage_df[
        [
            "region",
            "class_count",
            "classes_present"
        ]
    ],
    on="region",
    how="left"
)


# ============================================================
# 5. 增加审查指标
# ============================================================

# 是否覆盖全部类别
max_class_count = review_df["class_count"].max()

review_df["all_classes_present"] = (
    review_df["class_count"] == max_class_count
).map({
    True: "是",
    False: "否"
})


# 有效样本占原始 image 数量比例
review_df["valid_pair_ratio"] = (
    review_df["valid_pairs"]
    / review_df["images"]
)


# ============================================================
# 6. 根据：
#    第一优先：类别覆盖
#    第二优先：有效样本数量
# ============================================================

review_df = review_df.sort_values(
    by=[
        "all_classes_present",
        "valid_pairs"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# ============================================================
# 7. 添加审查编号
# ============================================================

review_df.insert(
    0,
    "review_rank",
    range(
        1,
        len(review_df) + 1
    )
)


# ============================================================
# 8. 保存完整审查表
# ============================================================

review_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 控制台只显示简洁结果
# ============================================================

print("\n")
print("=" * 60)
print("OOD 候选地区审查表")
print("=" * 60)

print(
    review_df[
        [
            "review_rank",
            "region",
            "valid_pairs",
            "class_count",
            "classes_present",
            "all_classes_present",
            "valid_pair_ratio"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. 汇总
# ============================================================

print("\n")
print("=" * 60)
print("审查汇总")
print("=" * 60)

print(
    f"当前 OOD 候选地区：{len(review_df)} 个"
)

print(
    f"类别数量最大值：{max_class_count}"
)

print(
    "覆盖全部已发现类别的候选地区：",
    int(
        (
            review_df["all_classes_present"] == "是"
        ).sum()
    )
)

print(
    "类别不完整的候选地区：",
    int(
        (
            review_df["all_classes_present"] == "否"
        ).sum()
    )
)


print("\n")
print("审查表已保存：")
print(OUTPUT_PATH)

print("\n进程结束。")