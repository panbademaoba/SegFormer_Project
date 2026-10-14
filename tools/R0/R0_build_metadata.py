from pathlib import Path
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

# OpenEarthMap 数据根目录
DATA_ROOT = Path(
    r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_XBD"
)

# 输出目录
OUTPUT_DIR = Path(
    r"/work_dirs/R0_audit"
)

OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"


# ============================================================
# 2. 检查路径
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"找不到数据目录:\n{DATA_ROOT}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 扫描所有地区
# ============================================================

records = []

regions = sorted(
    [
        p for p in DATA_ROOT.iterdir()
        if p.is_dir()
    ]
)


print("=" * 60)
print("开始生成 OpenEarthMap metadata")
print("=" * 60)

print(
    f"发现地区数量: {len(regions)}"
)


# ============================================================
# 4. 遍历地区
# ============================================================

for region_dir in regions:

    region = region_dir.name

    image_dir = region_dir / "images"
    label_dir = region_dir / "labels"


    # 检查目录
    if not image_dir.exists():
        continue

    if not label_dir.exists():
        continue


    # 获取所有影像
    images = sorted(
        image_dir.glob("*")
    )


    valid_count = 0


    for image_path in images:

        # 获取文件名
        stem = image_path.stem


        # 常见标签格式
        possible_labels = [
            label_dir / f"{stem}.png",
            label_dir / f"{stem}.tif",
            label_dir / f"{stem}.jpg"
        ]


        label_path = None


        for p in possible_labels:
            if p.exists():
                label_path = p
                break


        # 没有对应标签跳过
        if label_path is None:
            continue


        records.append(
            {
                "region": region,
                "image_path": str(
                    image_path
                ),
                "label_path": str(
                    label_path
                )
            }
        )


        valid_count += 1


    print(
        f"{region:<25}"
        f"{valid_count:>5} pairs"
    )


# ============================================================
# 5. 保存 metadata.csv
# ============================================================

df = pd.DataFrame(records)


if len(df) == 0:
    raise RuntimeError(
        "没有找到任何有效 image-label pair，请检查数据路径。"
    )


df = df.sort_values(
    by=[
        "region",
        "image_path"
    ]
)


df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 输出统计
# ============================================================

print("\n" + "=" * 60)
print("metadata生成完成")
print("=" * 60)


print(
    f"有效 image-label 对数量: {len(df)}"
)

print(
    f"涉及地区数量: {df['region'].nunique()}"
)


print(
    f"输出文件:\n{OUTPUT_CSV}"
)


print("\n前5行:")
print(df.head())