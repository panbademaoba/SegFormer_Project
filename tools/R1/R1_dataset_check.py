from pathlib import Path
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(r"D:\SegFormer_Project")

AUDIT_DIR = PROJECT_ROOT / "work_dirs" / "R0_audit"

TRAIN_LIST = AUDIT_DIR / "train_pool.txt"
VAL_LIST = AUDIT_DIR / "iid_val.txt"


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def get_label_path(image_path):
    image_path = Path(image_path)

    label_dir = image_path.parent.parent / "labels"

    label_path = label_dir / image_path.name

    if label_path.exists():
        return label_path

    return None


def check(data, name):

    missing_image = 0
    missing_label = 0
    size_error = 0

    classes = set()

    for img_path in data:

        img_path = Path(img_path)

        if not img_path.exists():
            missing_image += 1
            continue

        label_path = get_label_path(img_path)

        if label_path is None:
            missing_label += 1
            continue


        img = Image.open(img_path)
        label = Image.open(label_path)


        if img.size != label.size:
            size_error += 1


        classes.update(
            np.unique(
                np.array(label)
            ).tolist()
        )


    print(name)

    print("samples:", len(data))
    print("missing image:", missing_image)
    print("missing label:", missing_label)
    print("size mismatch:", size_error)
    print("classes:", sorted(classes))


train_data = read_txt(TRAIN_LIST)
val_data = read_txt(VAL_LIST)


check(train_data, "Train Pool")
print()

check(val_data, "IID Validation")