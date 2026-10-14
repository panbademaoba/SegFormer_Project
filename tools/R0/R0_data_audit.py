import os
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

# ============================================================
# 新增：Matplotlib 可视化
# ============================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# R0 数据审计总程序
# ============================================================


# ------------------------------------------------------------
# 1. OpenEarthMap 数据集位置
# ------------------------------------------------------------

DATA_ROOT = Path(
    r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD"
)


# 项目输出目录
OUTPUT_DIR = Path(
    r"/work_dirs/R0_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. OpenEarthMap 8 类
# ------------------------------------------------------------
#
# 说明：
# OpenEarthMap 官方有 8 个语义类别。
#
# 这里按照你当前数据中的像素值：
# 0 = background
# 1~8 = 8 个语义类别
#
# 官方颜色：
# Bareland       #800000
# Rangeland      #00FF24
# Developed      #949494
# Road           #FFFFFF
# Tree           #226126
# Water          #0045FF
# Agriculture    #4BB549
# Building       #DE1F07
#
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "Background",
    1: "Bareland",
    2: "Rangeland",
    3: "Developed space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture land",
    8: "Building",
}


CLASS_COLORS = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 255, 36),
    3: (148, 148, 148),
    4: (255, 255, 255),
    5: (34, 97, 38),
    6: (0, 69, 255),
    7: (75, 181, 73),
    8: (222, 31, 7),
}


VALID_CLASS_IDS = set(CLASS_NAMES.keys())


# ------------------------------------------------------------
# 3. 抽样设置
# ------------------------------------------------------------

RANDOM_SEED = 2026

# 指南要求至少抽查 50 对
SAMPLE_NUMBER = 50

random.seed(RANDOM_SEED)


# ============================================================
# 工具函数
# ============================================================


def md5_file(path, chunk_size=1024 * 1024):
    """
    计算文件 MD5，用于检查完全重复文件。
    """

    md5 = hashlib.md5()

    try:

        with open(path, "rb") as f:

            while True:

                data = f.read(chunk_size)

                if not data:
                    break

                md5.update(data)

        return md5.hexdigest()

    except Exception:

        return None


def get_tif_files(folder):
    """
    获取目录中的 tif 文件。
    """

    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".tif"
    )


def safe_read_label(label_path):
    """
    读取标签。

    返回：
        label
        error
    """

    try:

        with rasterio.open(label_path) as src:

            label = src.read(1)

        return label, None

    except Exception as e:

        return None, str(e)


def check_label_values(label):
    """
    检查标签中的类别 ID。
    """

    values = np.unique(label)

    invalid = [
        int(v)
        for v in values
        if int(v) not in VALID_CLASS_IDS
    ]

    return values, invalid


# ============================================================
# 新增：可视化函数
# ============================================================


def set_chinese_font():
    """
    设置 Matplotlib 字体。

    如果电脑有中文字体，则优先使用常见中文字体。
    """

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]

    plt.rcParams["axes.unicode_minus"] = False


def save_region_data_count_figure(region_df):
    """
    图1：
    各地区有效 image-label pair 数量统计。

    输出：
        fig1_region_data_count.png
    """

    print("\n正在生成图1：各地区数据量统计图...")

    if region_df.empty:
        print("region_df 为空，跳过图1。")
        return

    # 按有效 image-label pair 从大到小排序
    plot_df = region_df.sort_values(
        "valid_pairs",
        ascending=False
    ).copy()

    # 根据地区数量自动调整图像高度
    height = max(
        6,
        len(plot_df) * 0.28
    )

    plt.figure(
        figsize=(12, height)
    )

    plt.barh(
        plot_df["region"],
        plot_df["valid_pairs"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Valid image-label pairs"
    )

    plt.ylabel(
        "Region"
    )

    plt.title(
        "OpenEarthMap Multi-region Data Audit"
    )

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "fig1_region_data_count.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("图1已保存：")
    print(output_path)


def save_class_distribution_figure(class_df):
    """
    图2：
    各类别总体像素数量统计。

    输出：
        fig2_class_pixel_distribution.png
    """

    print("\n正在生成图2：类别像素分布图...")

    if class_df.empty:
        print("class_df 为空，跳过图2。")
        return

    plot_df = class_df.copy()

    # 按 class_id 排序
    plot_df = plot_df.sort_values(
        "class_id"
    )

    labels = plot_df[
        "class_name"
    ].tolist()

    values = plot_df[
        "pixel_count"
    ].tolist()

    plt.figure(
        figsize=(12, 7)
    )

    bars = plt.bar(
        range(len(labels)),
        values
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=35,
        ha="right"
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Pixel count"
    )

    plt.title(
        "OpenEarthMap Global Class Pixel Distribution"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # 柱状图顶部显示具体像素数量
    max_value = max(values) if values else 0

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + max_value * 0.01,

            f"{value:,}",

            ha="center",
            va="bottom",

            fontsize=8
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "fig2_class_pixel_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("图2已保存：")
    print(output_path)


def save_region_class_heatmap(region_class_df):
    """
    图3：
    Region × Class 类别覆盖矩阵。

    含义：
        1 = 该区域存在该类别
        0 = 该区域没有该类别

    为了便于周报阅读：
        有该类别 → 显示 ✓
        没有该类别 → 留空

    输出：
        fig3_region_class_heatmap.png
    """

    print(
        "\n正在生成图3："
        "Region × Class 类别覆盖矩阵..."
    )

    if region_class_df.empty:
        print(
            "region_class_df 为空，"
            "跳过图3。"
        )
        return

    plot_df = region_class_df.copy()

    # --------------------------------------------------------
    # 1. 获取 class_0 ~ class_8
    # --------------------------------------------------------

    class_columns = [
        f"class_{class_id}"
        for class_id in range(9)
        if f"class_{class_id}" in plot_df.columns
    ]

    if not class_columns:

        print(
            "没有找到 class_* 列，"
            "跳过图3。"
        )

        return

    # --------------------------------------------------------
    # 2. 转换为 Region × Class 矩阵
    # --------------------------------------------------------

    matrix = plot_df.set_index(
        "region"
    )[class_columns]

    # --------------------------------------------------------
    # 3. 转换为类别覆盖信息
    #
    # 像素数量 > 0：
    #     说明该区域存在这个类别
    #
    # 像素数量 = 0：
    #     说明该区域没有这个类别
    # --------------------------------------------------------

    coverage_matrix = (
        matrix > 0
    ).astype(int)

    # --------------------------------------------------------
    # 4. 按地区名称排序
    # --------------------------------------------------------

    coverage_matrix = (
        coverage_matrix.sort_index()
    )

    # --------------------------------------------------------
    # 5. 获取类别名称
    # --------------------------------------------------------

    class_ids = [
        int(
            col.split("_")[1]
        )
        for col in coverage_matrix.columns
    ]

    class_labels = [
        CLASS_NAMES[class_id]
        for class_id in class_ids
    ]

    # --------------------------------------------------------
    # 6. 设置图像高度
    # --------------------------------------------------------

    height = max(
        8,
        len(coverage_matrix.index) * 0.28
    )

    plt.figure(
        figsize=(12, height)
    )

    # --------------------------------------------------------
    # 7. 绘制二值覆盖矩阵
    # --------------------------------------------------------

    plt.imshow(
        coverage_matrix.values,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        cmap="Blues"
    )

    # --------------------------------------------------------
    # 8. 坐标轴
    # --------------------------------------------------------

    plt.xticks(
        range(
            len(class_labels)
        ),
        class_labels,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        range(
            len(coverage_matrix.index)
        ),
        coverage_matrix.index
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Region"
    )

    plt.title(
        "OpenEarthMap Region × Class Coverage"
    )

    # --------------------------------------------------------
    # 9. 在存在类别的位置添加 ✓
    # --------------------------------------------------------

    for row in range(
        coverage_matrix.shape[0]
    ):

        for col in range(
            coverage_matrix.shape[1]
        ):

            value = coverage_matrix.iloc[
                row,
                col
            ]

            if value == 1:

                plt.text(
                    col,
                    row,
                    "✓",
                    ha="center",
                    va="center",
                    fontsize=7
                )

    # --------------------------------------------------------
    # 10. 添加网格线
    # --------------------------------------------------------

    ax = plt.gca()

    ax.set_xticks(
        np.arange(
            -0.5,
            len(class_labels),
            1
        ),
        minor=True
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(coverage_matrix.index),
            1
        ),
        minor=True
    )

    ax.grid(
        which="minor",
        linestyle="-",
        linewidth=0.5,
        alpha=0.3
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # --------------------------------------------------------
    # 11. 添加图例
    # --------------------------------------------------------

    legend_elements = [
        Patch(
            label="1 = Class present"
        ),
        Patch(
            label="0 = Class absent"
        )
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # 12. 保存图3
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "fig3_region_class_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("图3已保存：")
    print(output_path)


def generate_visualizations(
    region_df,
    class_df,
    region_class_df
):
    """
    统一生成 R0 可视化结果。
    """

    print("\n" + "=" * 70)
    print("[14] 生成 R0 可视化结果")
    print("=" * 70)

    set_chinese_font()

    # 图1：各区域数据量
    save_region_data_count_figure(
        region_df
    )

    # 图2：总体类别像素分布
    save_class_distribution_figure(
        class_df
    )

    # 图3：区域 × 类别覆盖矩阵
    save_region_class_heatmap(
        region_class_df
    )

    print(
        "\nR0 可视化生成完成！"
    )


# ============================================================
# 主程序
# ============================================================


def main():

    print("=" * 70)
    print("R0 DATA AUDIT")
    print("=" * 70)

    print("\n数据根目录：")
    print(DATA_ROOT)

    # --------------------------------------------------------
    # 1. 基本目录检查
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[1] 数据目录检查")
    print("=" * 70)

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"找不到数据集：\n{DATA_ROOT}"
        )

    print(
        "数据集目录存在：是"
    )

    # 获取地区
    regions = sorted(
        p
        for p in DATA_ROOT.iterdir()
        if p.is_dir()
    )

    print(
        "发现地区数量：",
        len(regions)
    )

    # --------------------------------------------------------
    # 2. 统计每个地区影像、标签
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] 地区 / 影像 / 标签统计")
    print("=" * 70)

    region_records = []

    all_image_paths = []
    all_label_paths = []

    for region_dir in regions:

        region = region_dir.name

        image_dir = (
            region_dir
            / "images"
        )

        label_dir = (
            region_dir
            / "labels"
        )

        images = get_tif_files(
            image_dir
        )

        labels = get_tif_files(
            label_dir
        )

        image_names = {
            p.stem
            for p in images
        }

        label_names = {
            p.stem
            for p in labels
        }

        valid_names = (
            image_names
            & label_names
        )

        missing_labels = (
            image_names
            - label_names
        )

        missing_images = (
            label_names
            - image_names
        )

        region_records.append(
            {
                "region": region,
                "images": len(images),
                "labels": len(labels),
                "missing_labels": len(
                    missing_labels
                ),
                "missing_images": len(
                    missing_images
                ),
                "valid_pairs": len(
                    valid_names
                ),
            }
        )

        all_image_paths.extend(
            images
        )

        all_label_paths.extend(
            labels
        )

        print(
            f"{region:20s} "
            f"images={len(images):4d} "
            f"labels={len(labels):4d} "
            f"pairs={len(valid_names):4d}"
        )

    region_df = pd.DataFrame(
        region_records
    )

    # 保存地区统计
    region_csv = (
        OUTPUT_DIR
        / "dataset_report.csv"
    )

    region_df.to_csv(
        region_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n地区统计已保存："
    )

    print(
        region_csv
    )

    # --------------------------------------------------------
    # 3. 数据集总体统计
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] 数据集总体统计")
    print("=" * 70)

    total_regions = len(
        regions
    )

    total_images = len(
        all_image_paths
    )

    total_labels = len(
        all_label_paths
    )

    total_valid_pairs = int(
        region_df[
            "valid_pairs"
        ].sum()
    )

    print(
        "地区数量：",
        total_regions
    )

    print(
        "影像总数：",
        total_images
    )

    print(
        "标签总数：",
        total_labels
    )

    print(
        "有效 image-label pair：",
        total_valid_pairs
    )

    print(
        "\nOpenEarthMap 官方数据参考："
    )

    print(
        "官方地区数量：97"
    )

    print(
        "官方国家数量：44"
    )

    print(
        "官方语义类别：8"
    )

    # --------------------------------------------------------
    # 4. 国家数量
    # --------------------------------------------------------

    print("\n国家数量：")

    print(
        "本地目录没有 country 字段，"
        "因此不从 region 名称推测国家。"
    )

    print(
        "OpenEarthMap 官方国家数量：44"
    )

    # --------------------------------------------------------
    # 5. 每个地区影像数量
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] 每个地区影像数量")
    print("=" * 70)

    for _, row in region_df.iterrows():

        print(
            f"{row['region']:20s}"
            f"{int(row['images']):4d}"
        )

    # --------------------------------------------------------
    # 6. 标签类别像素统计
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[5] 8 类像素统计")
    print("=" * 70)

    global_class_counts = Counter()

    region_class_counts = defaultdict(
        lambda: Counter()
    )

    abnormal_labels = []

    corrupted_labels = []

    total_label_pixels = 0

    # 遍历所有标签
    for index, label_path in enumerate(
        all_label_paths
    ):

        region = (
            label_path
            .parent
            .parent
            .name
        )

        label, error = safe_read_label(
            label_path
        )

        if error is not None:

            corrupted_labels.append(
                (
                    str(label_path),
                    error
                )
            )

            continue

        # 标签尺寸
        total_label_pixels += label.size

        # 类别 ID 检查
        values, invalid = check_label_values(
            label
        )

        if invalid:

            abnormal_labels.append(
                {
                    "region": region,
                    "label": label_path.name,
                    "invalid_values": str(
                        invalid
                    ),
                }
            )

        # 像素统计
        counts = np.bincount(
            label.flatten(),
            minlength=9
        )

        for class_id in range(9):

            count = int(
                counts[class_id]
            )

            global_class_counts[
                class_id
            ] += count

            region_class_counts[
                region
            ][class_id] += count

    # 打印总类别统计

    print(
        f"{'ID':>3s} "
        f"{'类别':25s} "
        f"{'像素数量':>15s}"
    )

    for class_id in range(9):

        print(
            f"{class_id:3d} "
            f"{CLASS_NAMES[class_id]:25s} "
            f"{global_class_counts[class_id]:15d}"
        )

    # --------------------------------------------------------
    # 7. 每个地区的类别像素统计
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] 每个地区的类别像素统计")
    print("=" * 70)

    region_class_records = []

    for region in sorted(
        region_class_counts
    ):

        record = {
            "region": region
        }

        print(
            "\n地区：",
            region
        )

        for class_id in range(9):

            count = (
                region_class_counts[
                    region
                ][class_id]
            )

            record[
                f"class_{class_id}"
            ] = count

            print(
                f"  {class_id} "
                f"{CLASS_NAMES[class_id]:25s}"
                f": {count}"
            )

        region_class_records.append(
            record
        )

    region_class_df = pd.DataFrame(
        region_class_records
    )

    region_class_csv = (
        OUTPUT_DIR
        / "region_class_distribution.csv"
    )

    region_class_df.to_csv(
        region_class_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n地区类别统计已保存："
    )

    print(
        region_class_csv
    )

    # --------------------------------------------------------
    # 8. 全部类别统计保存
    # --------------------------------------------------------

    class_records = []

    for class_id in range(9):

        class_records.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "pixel_count":
                    global_class_counts[
                        class_id
                    ],
                "hex_color":
                    "#{:02X}{:02X}{:02X}".format(
                        *CLASS_COLORS[
                            class_id
                        ]
                    )
            }
        )

    class_df = pd.DataFrame(
        class_records
    )

    class_csv = (
        OUTPUT_DIR
        / "class_distribution.csv"
    )

    class_df.to_csv(
        class_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n总类别统计已保存："
    )

    print(
        class_csv
    )

    # --------------------------------------------------------
    # 9. 至少抽查 50 对 image-label
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[7] image-label 抽样检查")
    print("=" * 70)

    valid_pairs = []

    for region_dir in regions:

        region = region_dir.name

        image_dir = (
            region_dir
            / "images"
        )

        label_dir = (
            region_dir
            / "labels"
        )

        images = get_tif_files(
            image_dir
        )

        for image_path in images:

            label_path = (
                label_dir
                / image_path.name
            )

            if label_path.exists():

                valid_pairs.append(
                    (
                        region,
                        image_path,
                        label_path
                    )
                )

    print(
        "全部有效 image-label pair：",
        len(valid_pairs)
    )

    sample_number = min(
        SAMPLE_NUMBER,
        len(valid_pairs)
    )

    random.seed(
        RANDOM_SEED
    )

    sampled_pairs = random.sample(
        valid_pairs,
        sample_number
    )

    print(
        "本次实际抽查数量：",
        sample_number
    )

    sample_records = []

    spatial_errors = []

    color_mapping_errors = []

    sample_corrupted = []

    for (
        region,
        image_path,
        label_path
    ) in sampled_pairs:

        record = {
            "region": region,
            "image": image_path.name,
            "label": label_path.name,
        }

        try:

            with rasterio.open(
                image_path
            ) as image_src:

                image_shape = (
                    image_src.height,
                    image_src.width
                )

                image_count = (
                    image_src.count
                )

                image_crs = (
                    image_src.crs
                )

                image_transform = (
                    image_src.transform
                )

                # 真正读取数据，检查文件是否损坏
                image_data = (
                    image_src.read()
                )

            with rasterio.open(
                label_path
            ) as label_src:

                label_shape = (
                    label_src.height,
                    label_src.width
                )

                label_crs = (
                    label_src.crs
                )

                label_transform = (
                    label_src.transform
                )

                label_data = (
                    label_src.read(1)
                )

            record[
                "image_size"
            ] = str(
                image_shape
            )

            record[
                "label_size"
            ] = str(
                label_shape
            )

            record[
                "image_bands"
            ] = image_count

            # ------------------------------------------------
            # 空间尺寸检查
            # ------------------------------------------------

            size_match = (
                image_shape
                == label_shape
            )

            record[
                "size_match"
            ] = size_match

            # ------------------------------------------------
            # CRS 检查
            # ------------------------------------------------

            crs_match = (
                image_crs
                == label_crs
            )

            record[
                "crs_match"
            ] = crs_match

            # ------------------------------------------------
            # transform 检查
            # ------------------------------------------------

            transform_match = (
                image_transform
                == label_transform
            )

            record[
                "transform_match"
            ] = transform_match

            if not (
                size_match
                and crs_match
                and transform_match
            ):

                spatial_errors.append(
                    record.copy()
                )

            # ------------------------------------------------
            # 标签类别 ID 检查
            # ------------------------------------------------

            label_values = np.unique(
                label_data
            )

            invalid_values = [
                int(v)
                for v in label_values
                if int(v)
                not in VALID_CLASS_IDS
            ]

            record[
                "label_values"
            ] = str(
                [
                    int(v)
                    for v in label_values
                ]
            )

            record[
                "invalid_label_values"
            ] = str(
                invalid_values
            )

            # ------------------------------------------------
            # 颜色映射检查
            # ------------------------------------------------

            missing_color_ids = [
                int(v)
                for v in label_values
                if int(v)
                not in CLASS_COLORS
            ]

            record[
                "color_mapping_complete"
            ] = (
                len(
                    missing_color_ids
                ) == 0
            )

            if missing_color_ids:

                color_mapping_errors.append(
                    {
                        "region": region,
                        "label": label_path.name,
                        "missing_color_ids":
                            str(
                                missing_color_ids
                            )
                    }
                )

            sample_records.append(
                record
            )

        except Exception as e:

            sample_corrupted.append(
                {
                    "region": region,
                    "image": image_path.name,
                    "label": label_path.name,
                    "error": str(e)
                }
            )

    sample_df = pd.DataFrame(
        sample_records
    )

    sample_csv = (
        OUTPUT_DIR
        / "sample_check_report.csv"
    )

    sample_df.to_csv(
        sample_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n抽查报告已保存："
    )

    print(
        sample_csv
    )

    # --------------------------------------------------------
    # 10. 空间对齐检查
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[8] 空间对齐检查")
    print("=" * 70)

    print(
        "抽查数量：",
        sample_number
    )

    print(
        "空间不一致数量：",
        len(spatial_errors)
    )

    if spatial_errors:

        print(
            "\n发现空间不一致："
        )

        for item in spatial_errors:

            print(
                item["region"],
                item["image"]
            )

    else:

        print(
            "抽查样本中："
            "尺寸、CRS、Transform 均一致"
        )

    # --------------------------------------------------------
    # 11. 标签异常检查
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[9] 标签类别 ID 检查")
    print("=" * 70)

    print(
        "发现异常标签文件：",
        len(abnormal_labels)
    )

    if abnormal_labels:

        for item in abnormal_labels[:20]:

            print(
                item["region"],
                item["label"],
                item["invalid_values"]
            )

    else:

        print(
            "没有发现非法类别 ID"
        )

    # --------------------------------------------------------
    # 12. 颜色映射检查
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[10] 颜色映射检查")
    print("=" * 70)

    print(
        "OpenEarthMap 8 类官方颜色映射："
    )

    for class_id in range(1, 9):

        rgb = CLASS_COLORS[
            class_id
        ]

        hex_color = (
            "#{:02X}{:02X}{:02X}"
            .format(*rgb)
        )

        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]:20s}"
            f"{hex_color}"
        )

    print(
        "\n抽查样本中颜色映射缺失：",
        len(color_mapping_errors)
    )

    if color_mapping_errors:

        for item in color_mapping_errors:

            print(item)

    else:

        print(
            "抽查样本中的类别 ID "
            "均存在对应颜色映射"
        )

    # --------------------------------------------------------
    # 13. 损坏文件检查
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[11] 文件损坏检查")
    print("=" * 70)

    corrupted_images = []

    corrupted_labels = []

    # 检查所有 image
    for image_path in all_image_paths:

        try:

            with rasterio.open(
                image_path
            ) as src:

                src.read()

        except Exception as e:

            corrupted_images.append(
                {
                    "file": str(
                        image_path
                    ),
                    "error": str(e)
                }
            )

    # 检查所有 label
    for label_path in all_label_paths:

        try:

            with rasterio.open(
                label_path
            ) as src:

                src.read()

        except Exception as e:

            corrupted_labels.append(
                {
                    "file": str(
                        label_path
                    ),
                    "error": str(e)
                }
            )

    print(
        "损坏影像数量：",
        len(corrupted_images)
    )

    print(
        "损坏标签数量：",
        len(corrupted_labels)
    )

    # --------------------------------------------------------
    # 14. 重复文件检查
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[12] 重复文件检查")
    print("=" * 70)

    image_hashes = defaultdict(
        list
    )

    label_hashes = defaultdict(
        list
    )

    # 影像 MD5
    for image_path in all_image_paths:

        md5 = md5_file(
            image_path
        )

        if md5:

            image_hashes[
                md5
            ].append(
                str(image_path)
            )

    # 标签 MD5
    for label_path in all_label_paths:

        md5 = md5_file(
            label_path
        )

        if md5:

            label_hashes[
                md5
            ].append(
                str(label_path)
            )

    duplicate_images = {
        md5: paths
        for md5, paths
        in image_hashes.items()
        if len(paths) > 1
    }

    duplicate_labels = {
        md5: paths
        for md5, paths
        in label_hashes.items()
        if len(paths) > 1
    }

    print(
        "重复影像组数量：",
        len(duplicate_images)
    )

    print(
        "重复标签组数量：",
        len(duplicate_labels)
    )

    # --------------------------------------------------------
    # 15. 生成最终 R0 审计总结
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[13] 生成 R0 审计总结")
    print("=" * 70)

    summary_path = (
        OUTPUT_DIR
        / "R0_audit_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "OpenEarthMap R0 DATA AUDIT SUMMARY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"数据目录：{DATA_ROOT}\n\n"
        )

        f.write(
            f"本地地区数量："
            f"{total_regions}\n"
        )

        f.write(
            f"本地影像数量："
            f"{total_images}\n"
        )

        f.write(
            f"本地标签数量："
            f"{total_labels}\n"
        )

        f.write(
            f"有效 image-label pair："
            f"{total_valid_pairs}\n"
        )

        f.write(
            "OpenEarthMap 官方国家数量：44\n"
        )

        f.write(
            "OpenEarthMap 官方地区数量：97\n"
        )

        f.write(
            "OpenEarthMap 官方类别数量：8\n\n"
        )

        f.write(
            "类别像素统计\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for class_id in range(9):

            f.write(
                f"{class_id} "
                f"{CLASS_NAMES[class_id]}: "
                f"{global_class_counts[class_id]}\n"
            )

        f.write("\n")

        f.write(
            f"抽查 image-label 数量："
            f"{sample_number}\n"
        )

        f.write(
            f"空间对齐异常："
            f"{len(spatial_errors)}\n"
        )

        f.write(
            f"异常标签文件："
            f"{len(abnormal_labels)}\n"
        )

        f.write(
            f"损坏影像："
            f"{len(corrupted_images)}\n"
        )

        f.write(
            f"损坏标签："
            f"{len(corrupted_labels)}\n"
        )

        f.write(
            f"重复影像组："
            f"{len(duplicate_images)}\n"
        )

        f.write(
            f"重复标签组："
            f"{len(duplicate_labels)}\n"
        )

    # ========================================================
    # 16. 新增：生成可视化结果
    # ========================================================

    generate_visualizations(
        region_df=region_df,
        class_df=class_df,
        region_class_df=region_class_df
    )

    # --------------------------------------------------------
    # 17. 最终输出
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("R0 审计完成！")
    print("=" * 70)

    print(
        "\n最终报告："
    )

    print(
        summary_path
    )

    print(
        "\n主要输出文件："
    )

    print(
        "1.",
        OUTPUT_DIR
        / "dataset_report.csv"
    )

    print(
        "2.",
        OUTPUT_DIR
        / "class_distribution.csv"
    )

    print(
        "3.",
        OUTPUT_DIR
        / "region_class_distribution.csv"
    )

    print(
        "4.",
        OUTPUT_DIR
        / "sample_check_report.csv"
    )

    print(
        "5.",
        OUTPUT_DIR
        / "R0_audit_summary.txt"
    )

    print(
        "\n新增可视化文件："
    )

    print(
        "6.",
        OUTPUT_DIR
        / "fig1_region_data_count.png"
    )

    print(
        "7.",
        OUTPUT_DIR
        / "fig2_class_pixel_distribution.png"
    )

    print(
        "8.",
        OUTPUT_DIR
        / "fig3_region_class_heatmap.png"
    )

    print(
        "\n全部 R0 工作完成。"
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()