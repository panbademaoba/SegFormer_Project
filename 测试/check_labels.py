import os
import numpy as np
from PIL import Image
from collections import Counter

DATA_ROOT = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_xBD"


# ==========================
# 统计变量
# ==========================

class_counter = Counter()

total_pixels = 0

total_files = 0


# ==========================
# 遍历所有区域
# ==========================

for region in os.listdir(DATA_ROOT):

    region_path = os.path.join(DATA_ROOT, region)

    # 跳过非文件夹
    if not os.path.isdir(region_path):
        continue


    label_dir = os.path.join(region_path, "labels")


    # 没有labels跳过
    if not os.path.isdir(label_dir):
        continue


    print("\n正在检查:", region)


    for file in os.listdir(label_dir):

        # OpenEarthMap标签是tif
        if file.endswith(".tif"):

            label_path = os.path.join(label_dir, file)


            try:

                img = np.array(Image.open(label_path))


                # 打印前几个样本信息
                if total_files < 10:
                    print(
                        file,
                        "shape:",
                        img.shape,
                        "classes:",
                        np.unique(img)
                    )


                # 统计类别
                values, counts = np.unique(
                    img,
                    return_counts=True
                )


                for v, c in zip(values, counts):

                    class_counter[int(v)] += int(c)


                total_pixels += img.size

                total_files += 1


            except Exception as e:

                print(
                    "读取失败:",
                    label_path,
                    e
                )



# ==========================
# 输出结果
# ==========================


print("\n====================")
print("标签统计结果")
print("====================")


print(
    "标签文件数量:",
    total_files
)


print(
    "总像素数量:",
    total_pixels
)


print(
    "类别数量:",
    len(class_counter)
)


print("\n类别像素统计:")


for cls, count in sorted(class_counter.items()):

    ratio = count / total_pixels * 100

    print(
        "类别",
        cls,
        ":",
        count,
        "像素",
        "占比 %.2f%%" % ratio
    )