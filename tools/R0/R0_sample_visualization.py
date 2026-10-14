import os
import random
import csv

import rasterio
import numpy as np
import matplotlib.pyplot as plt


# ==================================
# OpenEarthMap路径
# ==================================

DATA_ROOT = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD"


# 输出路径

OUTPUT_DIR = r"/work_dirs/R0_audit"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==================================
# 参数
# ==================================

# 抽查数量

NUM_SAMPLES = 50


# 每张图显示几个样本

SHOW_PER_PAGE = 10



# ==================================
# 获取所有区域影像
# ==================================

samples = []


regions = os.listdir(DATA_ROOT)


for region in regions:


    image_dir = os.path.join(
        DATA_ROOT,
        region,
        "images"
    )


    label_dir = os.path.join(
        DATA_ROOT,
        region,
        "labels"
    )


    if not os.path.isdir(image_dir):
        continue


    images = [
        f for f in os.listdir(image_dir)
        if f.endswith(".tif")
    ]


    for img in images:

        label = os.path.join(
            label_dir,
            img
        )


        if os.path.exists(label):

            samples.append(
                {
                    "region": region,
                    "image": os.path.join(
                        image_dir,
                        img
                    ),
                    "label": label
                }
            )



print("==============================")
print("全部可抽查样本:", len(samples))



# ==================================
# 随机抽样
# ==================================

if len(samples) < NUM_SAMPLES:

    selected_samples = samples

else:

    selected_samples = random.sample(
        samples,
        NUM_SAMPLES
    )


print(
    "实际抽查数量:",
    len(selected_samples)
)



# ==================================
# CSV报告
# ==================================

report_file = os.path.join(
    OUTPUT_DIR,
    "sample_check_report.csv"
)



report = []



# ==================================
# 可视化
# ==================================

page = 1

count = 0


fig = None



for idx, item in enumerate(selected_samples):


    region = item["region"]

    image_path = item["image"]

    label_path = item["label"]



    # --------------------------
    # 读取image
    # --------------------------

    with rasterio.open(image_path) as src:

        image = src.read()




    image = np.transpose(
        image,
        (1,2,0)
    )



    # --------------------------
    # 读取label
    # --------------------------

    with rasterio.open(label_path) as src:

        label = src.read(1)




    # --------------------------
    # 尺寸检查
    # --------------------------

    height1, width1 = image.shape[:2]


    height2, width2 = label.shape



    size_match = (
        height1 == height2
        and
        width1 == width2
    )



    # --------------------------
    # 类别检查
    # --------------------------

    classes = np.unique(label)


    classes = ",".join(
        map(
            str,
            classes
        )
    )



    report.append(
        {
            "region": region,
            "image": os.path.basename(image_path),
            "image_size":
                f"{height1}x{width1}",
            "label_size":
                f"{height2}x{width2}",
            "size_match":
                size_match,
            "classes":
                classes
        }
    )



    # --------------------------
    # 创建可视化页面
    # --------------------------

    if count % SHOW_PER_PAGE == 0:


        if fig:

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    f"sample_visualization_{page}.png"
                ),
                dpi=200
            )


            plt.close()



            page += 1



        fig = plt.figure(
            figsize=(15,12)
        )



    position = (
        count % SHOW_PER_PAGE
    ) * 2 + 1



    # image

    plt.subplot(
        SHOW_PER_PAGE,
        2,
        position
    )


    plt.imshow(
        image[:,:,:3] / 255
    )


    plt.title(
        region
    )

    plt.axis("off")



    # label

    plt.subplot(
        SHOW_PER_PAGE,
        2,
        position + 1
    )


    plt.imshow(
        label,
        cmap="tab20"
    )


    plt.title(
        "Label"
    )

    plt.axis("off")



    count += 1




# 保存最后一页

if fig:


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"sample_visualization_{page}.png"
        ),
        dpi=200
    )


    plt.close()



# ==================================
# 保存CSV
# ==================================


with open(
    report_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=report[0].keys()
    )


    writer.writeheader()

    writer.writerows(
        report
    )



print("==============================")
print("抽查完成")

print(
    "报告:",
    report_file
)


print(
    "可视化保存:",
    OUTPUT_DIR
)
print("可视化完成")
