import os
import json
from collections import Counter

import numpy as np
from PIL import Image


# ============================================================
# R0-03：标签内容审计
# ============================================================

ROOT = r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_XBD\tokyo"
LABEL_DIR = os.path.join(ROOT, "labels")

OUT_DIR = r"/work_dirs/R0_audit"
OUT_FILE = os.path.join(OUT_DIR, "R0_label_audit.json")

os.makedirs(OUT_DIR, exist_ok=True)


print("R0 Label Audit")

print(f"标签目录：{LABEL_DIR}")
print(f"目录存在：{os.path.exists(LABEL_DIR)}")


# ------------------------------------------------------------
# 1. 获取标签文件
# ------------------------------------------------------------

label_files = sorted(
    [
        f for f in os.listdir(LABEL_DIR)
        if f.lower().endswith((".tif", ".tiff"))
    ]
)

print(f"\n标签文件数量：{len(label_files)}")


# ------------------------------------------------------------
# 2. 逐张检查标签
# ------------------------------------------------------------

global_counter = Counter()

empty_labels = []
abnormal_labels = []

label_details = {}

for filename in label_files:

    path = os.path.join(LABEL_DIR, filename)

    try:
        with Image.open(path) as img:

            arr = np.array(img)

            unique_values, counts = np.unique(
                arr,
                return_counts=True
            )

            value_counter = {
                int(v): int(c)
                for v, c in zip(unique_values, counts)
            }

            global_counter.update(value_counter)

            # 是否为空标签
            # 只有一个像素值，并且该值为 0
            is_empty = (
                len(unique_values) == 1
                and int(unique_values[0]) == 0
            )

            if is_empty:
                empty_labels.append(filename)

            # 当前标签的最大像素值
            max_value = int(np.max(arr))

            # OpenEarthMap 常规类别标签通常是有限的整数类别编号。
            # 这里先记录 > 255 的值作为明显异常，
            # 暂不擅自判断 0~255 中哪些具体类别正确。
            abnormal_values = [
                int(v)
                for v in unique_values
                if int(v) > 255
            ]

            if abnormal_values:
                abnormal_labels.append({
                    "file": filename,
                    "values": abnormal_values
                })

            label_details[filename] = {
                "size": list(img.size),
                "mode": img.mode,
                "unique_values": [int(v) for v in unique_values],
                "pixel_counts": value_counter,
                "empty_label": is_empty
            }

    except Exception as e:

        abnormal_labels.append({
            "file": filename,
            "error": str(e)
        })


# ------------------------------------------------------------
# 3. 汇总结果
# ------------------------------------------------------------

print("[1] 全部标签出现过的像素值")

all_values = sorted(global_counter.keys())

print(all_values)


print("[2] 各像素值的总像素数量")

for value in all_values:
    print(f"值 {value:>3} : {global_counter[value]}")


print("[3] 空标签检查")

print(f"空标签数量：{len(empty_labels)}")

if empty_labels:
    print("空标签：")
    for filename in empty_labels:
        print(" ", filename)
else:
    print("没有发现全黑空标签。")


print("[4] 异常值检查")

print(f"异常标签数量：{len(abnormal_labels)}")

if abnormal_labels:
    for item in abnormal_labels:
        print(item)
else:
    print("没有发现 >255 的像素值。")


print("[5] 每张标签的像素值")

for filename in label_files:

    if filename in label_details:

        values = label_details[filename]["unique_values"]

        print(
            f"{os.path.splitext(filename)[0]:<15} "
            f"{values}"
        )


# ------------------------------------------------------------
# 4. 保存 JSON 审计结果
# ------------------------------------------------------------

result = {
    "dataset": "OpenEarthMap / Tokyo",

    "label_count": len(label_files),

    "global_pixel_values": all_values,

    "global_pixel_counts": {
        str(k): int(v)
        for k, v in sorted(global_counter.items())
    },

    "empty_labels": empty_labels,

    "abnormal_labels": abnormal_labels,

    "label_details": label_details
}


with open(
    OUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )


print("R0 Label Audit 完成")

print(f"\n报告已保存：")
print(OUT_FILE)