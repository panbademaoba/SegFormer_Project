import os
import csv
from PIL import Image
from collections import Counter


# =========================
# 数据路径
# =========================

DATA_ROOT = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD"

SAVE_DIR = r"D:\SegFormer_Project\work_dirs\R0_audit"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 保存文件
# =========================

PAIR_CSV = os.path.join(
    SAVE_DIR,
    "region_pair_check.csv"
)

SIZE_CSV = os.path.join(
    SAVE_DIR,
    "region_size_check.csv"
)

CLASS_CSV = os.path.join(
    SAVE_DIR,
    "region_class_distribution.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "R0_audit_summary.txt"
)


# =========================
# 统计变量
# =========================

total_images = 0
total_labels = 0

class_counter = Counter()

size_error = []

pair_results = []

size_results = []



# =========================
# 遍历所有区域
# =========================

regions = os.listdir(DATA_ROOT)


for region in regions:


    region_path = os.path.join(
        DATA_ROOT,
        region
    )


    if not os.path.isdir(region_path):
        continue


    image_dir = os.path.join(
        region_path,
        "images"
    )


    label_dir = os.path.join(
        region_path,
        "labels"
    )


    if not os.path.exists(image_dir):
        continue


    if not os.path.exists(label_dir):
        continue



    print("检查地区:", region)



    # -------------------------
    # 获取文件
    # -------------------------

    image_files = set(
        [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".png",".tif",".tiff")
            )
        ]
    )


    label_files = set(
        [
            f for f in os.listdir(label_dir)
            if f.lower().endswith(
                (".png",".tif",".tiff")
            )
        ]
    )


    total_images += len(image_files)
    total_labels += len(label_files)



    # -------------------------
    # 图片标签对应
    # -------------------------

    missing_labels = image_files - label_files

    missing_images = label_files - image_files



    pair_results.append(
        [
            region,
            len(image_files),
            len(label_files),
            len(missing_labels),
            len(missing_images)
        ]
    )



    # -------------------------
    # 尺寸检查
    # -------------------------

    common_files = image_files & label_files


    for file in common_files:


        image_path = os.path.join(
            image_dir,
            file
        )


        label_path = os.path.join(
            label_dir,
            file
        )


        try:

            img = Image.open(
                image_path
            )

            label = Image.open(
                label_path
            )


            img_size = img.size

            label_size = label.size



            size_results.append(
                [
                    region,
                    file,
                    img_size,
                    label_size
                ]
            )



            if img_size != label_size:

                size_error.append(
                    [
                        region,
                        file,
                        img_size,
                        label_size
                    ]
                )


        except Exception as e:

            print(
                "读取失败:",
                image_path,
                e
            )



    # -------------------------
    # 标签类别统计
    # -------------------------

    for file in label_files:


        label_path = os.path.join(
            label_dir,
            file
        )


        try:

            label = Image.open(
                label_path
            )


            pixels = list(
                label.getdata()
            )


            class_counter.update(
                pixels
            )


        except:

            pass




# =========================
# 保存 CSV
# =========================


# 图片标签对应

with open(
    PAIR_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "region",
            "images",
            "labels",
            "missing_labels",
            "missing_images"
        ]
    )


    writer.writerows(
        pair_results
    )



# 尺寸检查

with open(
    SIZE_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:


    writer = csv.writer(f)


    writer.writerow(
        [
            "region",
            "file",
            "image_size",
            "label_size"
        ]
    )


    writer.writerows(
        size_results
    )



# 类别统计

total_pixels = sum(
    class_counter.values()
)


with open(
    CLASS_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:


    writer = csv.writer(f)


    writer.writerow(
        [
            "class",
            "pixels",
            "percentage"
        ]
    )


    for k in sorted(class_counter):


        percent = (
            class_counter[k]
            /
            total_pixels
            *
            100
        )


        writer.writerow(
            [
                k,
                class_counter[k],
                f"{percent:.2f}%"
            ]
        )



# =========================
# 总报告
# =========================


with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:


    f.write(
        "R0审计结果\n"
    )

    f.write(
        "================\n\n"
    )


    f.write(
        f"总images: {total_images}\n"
    )

    f.write(
        f"总labels: {total_labels}\n"
    )

    f.write(
        f"尺寸异常数量: {len(size_error)}\n"
    )

    f.write(
        f"类别数量: {len(class_counter)}\n\n"
    )


    f.write(
        "类别统计:\n"
    )


    for k in sorted(class_counter):

        f.write(
            f"class {k}: {class_counter[k]} pixels\n"
        )



# =========================
# 输出结果
# =========================


print("\n===================")
print("R0审计完成")
print("===================")


print(
    "总images:",
    total_images
)

print(
    "总labels:",
    total_labels
)

print(
    "尺寸异常数量:",
    len(size_error)
)

print(
    "类别数量:",
    len(class_counter)
)


print(
    "\n报告保存位置:"
)

print(
    SAVE_DIR
)