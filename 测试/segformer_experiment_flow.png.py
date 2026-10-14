import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# 保存路径
save_path = r"D:\SegFormer_Project\work_dirs\R0_audit\R2_R3_experiment_flow.png"


# 创建画布
fig, ax = plt.subplots(figsize=(8, 10))

ax.axis("off")


# 流程内容
steps = [
    "OpenEarthMap 数据集",

    "R0 数据审核与准备\n"
    "• 数据完整性检查\n"
    "• 类别分布统计\n"
    "• 影像-标签一致性验证",

    "区域空间划分（IID / OOD）\n"
    "Train Pool: 1437 samples\n"
    "IID Validation: 360 samples\n"
    "OOD Test: 890 samples",

    "R2 模型训练与性能评估\n"
    "SegFormer模型训练\n"
    "评价指标：mIoU / Accuracy",

    "R3 泛化能力评价\n"
    "IID vs OOD性能比较\n"
    "跨区域泛化能力分析"
]


# 框的位置
positions = [
    (0.25, 0.82),
    (0.25, 0.64),
    (0.25, 0.44),
    (0.25, 0.24),
    (0.25, 0.06)
]


# 绘制流程框
for text, (x, y) in zip(steps, positions):

    box = FancyBboxPatch(
        (x, y),
        0.5,
        0.12,
        boxstyle="round,pad=0.02",
        linewidth=1.5
    )

    ax.add_patch(box)

    ax.text(
        x + 0.25,
        y + 0.06,
        text,
        ha="center",
        va="center",
        fontsize=10
    )


# 绘制箭头
for i in range(len(positions)-1):

    x = 0.5

    start_y = positions[i][1]

    end_y = positions[i+1][1] + 0.12


    ax.annotate(
        "",
        xy=(x, end_y),
        xytext=(x, start_y),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.5
        )
    )


# 标题
plt.title(
    "R2/R3 Experiment Workflow Design",
    fontsize=14,
    pad=20
)


# 保存
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("图片已保存:")
print(save_path)