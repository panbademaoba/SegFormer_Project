import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================
# 1. 文件路径
# ==========================

csv_path = Path(
    r"D:\SegFormer_Project\work_dirs\R0_audit\iid_split_statistics.csv"
)

save_path = Path(
    r"D:\SegFormer_Project\work_dirs\R0_audit\region_train_iid_distribution.png"
)


# ==========================
# 2. 读取数据
# ==========================

df = pd.read_csv(csv_path)


# 每个区域总样本数量
df["total_count"] = df["train_count"] + df["iid_count"]


# 选择样本数量最多的15个区域
df_plot = (
    df.sort_values(
        by="total_count",
        ascending=False
    )
    .head(15)
)


# 为了让柱状图从小到大排列
df_plot = df_plot.sort_values(
    by="total_count"
)


# ==========================
# 3. 绘制堆叠柱状图
# ==========================

plt.figure(figsize=(10, 6))


# Train Pool
plt.barh(
    df_plot["region"],
    df_plot["train_count"],
    label="Train Pool"
)


# IID Validation
plt.barh(
    df_plot["region"],
    df_plot["iid_count"],
    left=df_plot["train_count"],
    label="IID Validation"
)


# ==========================
# 4. 图像设置
# ==========================

plt.xlabel(
    "Number of Samples"
)

plt.ylabel(
    "Region"
)

plt.title(
    "Regional Train/IID Sample Distribution after Spatial Split"
)


plt.legend()


# 添加数字标注
for i, row in df_plot.iterrows():

    total = row["train_count"] + row["iid_count"]

    plt.text(
        total + 1,
        list(df_plot["region"]).index(row["region"]),
        str(total),
        va="center"
    )


plt.tight_layout()


# ==========================
# 5. 保存
# ==========================

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("图片已保存：")
print(save_path)