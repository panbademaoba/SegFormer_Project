import pandas as pd
import matplotlib.pyplot as plt
import os


AUDIT_DIR = r"D:\SegFormer_Project\work_dirs\R0_audit"


df = pd.read_csv(
    os.path.join(
        AUDIT_DIR,
        "dataset_report.csv"
    )
)


plt.figure(figsize=(14,6))


plt.bar(
    df["region"],
    df["images"],
    label="images"
)


plt.bar(
    df["region"],
    df["labels"],
    label="labels"
)


plt.xticks(
    rotation=90
)

plt.xlabel(
    "Region"
)

plt.ylabel(
    "Number"
)

plt.title(
    "OpenEarthMap Region Image and Label Statistics"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    os.path.join(
        AUDIT_DIR,
        "region_distribution.png"
    ),
    dpi=300
)


plt.close()


print("region_distribution.png完成")