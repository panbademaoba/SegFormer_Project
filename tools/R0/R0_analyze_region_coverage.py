import os
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = r"D:\SegFormer_Project"

DATASET_ROOT = (
    r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD"
)

CANDIDATE_PATH = os.path.join(
    PROJECT_ROOT,
    "work_dirs",
    "R0_audit",
    "region_split_candidates.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "work_dirs",
    "R0_audit",
    "region_class_coverage.csv"
)


# OpenEarthMap 中常见的无效/忽略标签值
IGNORE_VALUES = {255}


# ============================================================
# 2. 检查路径
# ============================================================

print("=" * 60)
print("OpenEarthMap 地区类别覆盖分析")
print("=" * 60)

print("\n数据集路径：")
print(DATASET_ROOT)

print("\n候选地区表：")
print(CANDIDATE_PATH)


if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(
        f"\n找不到数据集目录：\n{DATASET_ROOT}"
    )


if not os.path.exists(CANDIDATE_PATH):
    raise FileNotFoundError(
        f"\n找不到候选地区表：\n{CANDIDATE_PATH}"
    )


# ============================================================
# 3. 读取候选地区表
# ============================================================

candidate_df = pd.read_csv(CANDIDATE_PATH)

print("\n候选地区表读取成功。")
print(f"候选表中的地区数量：{len(candidate_df)}")


# 获取目前被标记为候选的地区
candidate_regions = set(
    candidate_df.loc[
        candidate_df["ood_candidate_15_20pct"] == "候选",
        "region"
    ].astype(str)
)

print(f"当前 OOD 候选地区数量：{len(candidate_regions)}")


# ============================================================
# 4. 获取所有地区
# ============================================================

regions = []

for name in os.listdir(DATASET_ROOT):

    region_path = os.path.join(
        DATASET_ROOT,
        name
    )

    if not os.path.isdir(region_path):
        continue

    images_dir = os.path.join(
        region_path,
        "images"
    )

    labels_dir = os.path.join(
        region_path,
        "labels"
    )

    if not os.path.isdir(images_dir):
        continue

    if not os.path.isdir(labels_dir):
        continue

    regions.append(name)


regions = sorted(regions)

print("\n实际扫描到的地区数量：")
print(len(regions))


# ============================================================
# 5. 分析每个地区的类别覆盖
# ============================================================

results = []


for index, region in enumerate(regions, start=1):

    print(
        f"\n[{index}/{len(regions)}] "
        f"正在分析地区：{region}"
    )

    region_path = os.path.join(
        DATASET_ROOT,
        region
    )

    images_dir = os.path.join(
        region_path,
        "images"
    )

    labels_dir = os.path.join(
        region_path,
        "labels"
    )


    # --------------------------------------------------------
    # 获取 image / label 文件
    # --------------------------------------------------------

    image_files = {
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(".tif")
    }

    label_files = {
        f
        for f in os.listdir(labels_dir)
        if f.lower().endswith(".tif")
    }


    # 只分析真正存在 image-label pair 的样本
    valid_files = sorted(
        image_files & label_files
    )


    if len(valid_files) == 0:

        print("  没有有效 image-label pair，跳过。")

        results.append({
            "region": region,
            "valid_pairs": 0,
            "class_count": 0,
            "classes_present": "",
            "is_current_ood_candidate": (
                "候选"
                if region in candidate_regions
                else ""
            )
        })

        continue


    # --------------------------------------------------------
    # 统计该地区出现过的类别
    # --------------------------------------------------------

    region_classes = set()

    total_pixels = 0

    class_pixel_counts = {}


    for filename in valid_files:

        label_path = os.path.join(
            labels_dir,
            filename
        )

        try:

            with rasterio.open(label_path) as src:

                label = src.read(1)

        except Exception as e:

            print(
                f"  读取失败：{filename}"
            )

            print(
                f"  错误信息：{e}"
            )

            continue


        # 获取该 label 中出现的值
        unique_values, counts = np.unique(
            label,
            return_counts=True
        )


        for value, count in zip(
            unique_values,
            counts
        ):

            value = int(value)
            count = int(count)

            # 忽略 255 等无效值
            if value in IGNORE_VALUES:
                continue

            region_classes.add(value)

            class_pixel_counts[value] = (
                class_pixel_counts.get(value, 0)
                + count
            )

            total_pixels += count


    # --------------------------------------------------------
    # 类别排序
    # --------------------------------------------------------

    classes_sorted = sorted(
        region_classes
    )


    classes_text = ",".join(
        str(x)
        for x in classes_sorted
    )


    # --------------------------------------------------------
    # 计算每个类别是否出现
    # --------------------------------------------------------

    result = {
        "region": region,
        "valid_pairs": len(valid_files),
        "class_count": len(classes_sorted),
        "classes_present": classes_text,
        "is_current_ood_candidate": (
            "候选"
            if region in candidate_regions
            else ""
        )
    }


    # 增加每个类别的像素数量
    for class_id in sorted(class_pixel_counts):

        result[
            f"class_{class_id}_pixels"
        ] = class_pixel_counts[class_id]


    results.append(result)


# ============================================================
# 6. 转换为 DataFrame
# ============================================================

coverage_df = pd.DataFrame(
    results
)


# ============================================================
# 7. 按有效样本数量排序
# ============================================================

coverage_df = coverage_df.sort_values(
    by=[
        "is_current_ood_candidate",
        "valid_pairs"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# ============================================================
# 8. 保存结果
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

coverage_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 简洁输出
# ============================================================

print("\n")
print("=" * 60)
print("当前 OOD 候选地区的类别覆盖")
print("=" * 60)

candidate_result = coverage_df[
    coverage_df["is_current_ood_candidate"] == "候选"
].copy()

if len(candidate_result) > 0:

    print(
        candidate_result[
            [
                "region",
                "valid_pairs",
                "class_count",
                "classes_present"
            ]
        ].to_string(index=False)
    )

else:

    print("当前没有 OOD 候选地区。")


# ============================================================
# 10. 整个数据集类别统计
# ============================================================

all_classes = set()

for classes_text in coverage_df[
    "classes_present"
].dropna():

    if classes_text == "":
        continue

    for value in classes_text.split(","):

        if value.strip() != "":
            all_classes.add(int(value))


print("\n")
print("=" * 60)
print("整个数据集类别统计")
print("=" * 60)

print("出现的类别 ID：", sorted(all_classes))
print("总类别数量：", len(all_classes))


# ============================================================
# 11. 完成
# ============================================================

print("\n")
print("=" * 60)
print("分析完成")
print("=" * 60)

print("\n结果文件：")
print(OUTPUT_PATH)

print("\n进程结束。")