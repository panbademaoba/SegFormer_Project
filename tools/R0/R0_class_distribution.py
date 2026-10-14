import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# OpenEarthMap 数据路径
# ===============================

DATA_ROOT = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD"

OUTPUT_DIR = r"/work_dirs/R0_audit"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# 类别统计
# ===============================

class_count = {}


regions = os.listdir(DATA_ROOT)


for region in regions:

    region_path = os.path.join(DATA_ROOT, region)

    label_dir = os.path.join(region_path, "labels")


    if not os.path.isdir(label_dir):
        continue


    print("正在处理:", region)


    for label_file in os.listdir(label_dir):

        if not label_file.endswith(".tif"):
            continue


        label_path = os.path.join(label_dir, label_file)


        try:

            with rasterio.open(label_path) as src:
                label = src.read(1)


            classes, counts = np.unique(
                label,
                return_counts=True
            )


            for c, n in zip(classes, counts):

                c = int(c)

                if c not in class_count:
                    class_count[c] = 0

                class_count[c] += int(n)


        except Exception as e:

            print("读取失败:", label_path)
            print(e)



# ===============================
# 保存CSV
# ===============================

df = pd.DataFrame(
    {
        "class_id": list(class_count.keys()),
        "pixel_count": list(class_count.values())
    }
)


df = df.sort_values(
    by="class_id"
)


csv_path = os.path.join(
    OUTPUT_DIR,
    "class_distribution.csv"
)


df.to_csv(
    csv_path,
    index=False
)



# ===============================
# 绘制类别统计图
# ===============================

plt.figure(figsize=(10,6))


plt.bar(
    df["class_id"],
    df["pixel_count"]
)


plt.xlabel("Class ID")
plt.ylabel("Pixel Count")

plt.title(
    "OpenEarthMap Class Distribution"
)


plt.xticks(
    df["class_id"]
)


plt.tight_layout()



png_path = os.path.join(
    OUTPUT_DIR,
    "class_distribution.png"
)


plt.savefig(
    png_path,
    dpi=300
)


plt.close()



print("====================")
print("类别统计完成")
print(csv_path)
print(png_path)