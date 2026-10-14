import json
from pathlib import Path


# ============================================================
# R0 Metadata
# ============================================================

PROJECT_ROOT = Path(r"/")

OUTPUT_DIR = PROJECT_ROOT / "work_dirs" / "R0_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = {
    "audit_version": "R0",

    "dataset": {
        "name": "OpenEarthMap",
        "subset": "Tokyo",
        "root": r"D:\OpenEarthMap\OpenEarthMap\OpenEarthMap_wo_XBD\tokyo"
    },

    "image": {
        "count": 70,
        "size": [1024, 1024],
        "mode": "RGB",
        "format": "TIFF"
    },

    "label": {
        "count": 49,
        "size": [1024, 1024],
        "mode": "L",
        "format": "TIFF"
    },

    "image_label_relationship": {
        "images_with_labels": 49,
        "missing_labels": 21,
        "extra_labels": 0,
        "size_mismatch": 0
    },

    "official_split_files": {
        "train": 3000,
        "val": 500,
        "test": 1500,
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
        "total_split_entries": 5000,
        "tokyo_images_in_split": 70
    },

    "audit_status": {
        "directory_check": "PASS",
        "image_label_check": "PASS_WITH_MISSING_LABELS",
        "split_overlap_check": "PASS",
        "image_size_check": "PASS",
        "label_size_check": "PASS",
        "image_label_size_match": "PASS"
    }
}


output_file = OUTPUT_DIR / "R0_metadata.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=4
    )

print("=" * 60)
print("R0 Metadata")
print("=" * 60)

print("数据集：OpenEarthMap / Tokyo")
print("图像数量：70")
print("标签数量：49")
print("缺少标签：21")
print("图像尺寸：1024 × 1024")
print("图像模式：RGB")
print("标签模式：L")

print()
print("Metadata 已保存：")
print(output_file)

print()
print("R0 Metadata 完成")