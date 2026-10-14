import csv
import json
import os
from pathlib import Path

AUDIT_DIR = Path(r"D:\SegFormer_Project\work_dirs\R0_audit")


def normalize_path(value):
    return os.path.normpath(value.strip().replace("\\", "/"))


def path_key(value):
    return os.path.normcase(normalize_path(value))


def read_list(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [
            normalize_path(line)
            for line in f
            if line.strip()
        ]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    with (AUDIT_DIR / "split_region.json").open(
        "r", encoding="utf-8-sig"
    ) as f:
        split = json.load(f)

    with (AUDIT_DIR / "metadata.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    ood_regions = set(split["ood"])
    train_regions = set(split["train_candidates"])
    require(
        not ood_regions & train_regions,
        "OOD 地区与训练候选地区存在重叠。"
    )
    require(
        all(row["region"] in ood_regions | train_regions for row in rows),
        "metadata 中存在未纳入固定划分的地区。"
    )

    ood_rows = [
        row for row in rows
        if row["region"] in ood_regions
    ]
    ood_paths = [
        normalize_path(row["image_path"])
        for row in ood_rows
    ]

    train_paths = read_list(AUDIT_DIR / "train_pool.txt")
    iid_paths = read_list(AUDIT_DIR / "iid_val.txt")

    train_set = {path_key(p) for p in train_paths}
    iid_set = {path_key(p) for p in iid_paths}
    ood_set = {path_key(p) for p in ood_paths}
    metadata_set = {path_key(row["image_path"]) for row in rows}

    for name, paths, unique in [
        ("Train", train_paths, train_set),
        ("IID", iid_paths, iid_set),
        ("OOD", ood_paths, ood_set),
    ]:
        require(len(paths) == len(unique), f"{name} 清单存在重复。")

    require(len(metadata_set) == len(rows), "metadata 存在重复影像。")
    require(len(train_paths) == 1437, "Train 数量不是1437。")
    require(len(iid_paths) == 360, "IID 数量不是360。")
    require(len(ood_paths) == 890, "OOD 数量不是890。")
    require(not train_set & iid_set, "Train 与 IID 存在重叠。")
    require(not train_set & ood_set, "Train 与 OOD 存在重叠。")
    require(not iid_set & ood_set, "IID 与 OOD 存在重叠。")
    require(
        train_set | iid_set | ood_set == metadata_set,
        "三份清单未完整覆盖 metadata，或包含额外样本。"
    )

    missing = []
    for row in ood_rows:
        for field in ("image_path", "label_path"):
            path = normalize_path(row[field])
            if not Path(path).is_file():
                missing.append((field, path))

    if missing:
        for field, path in missing[:20]:
            print(f"缺失 {field}: {path}")
        raise RuntimeError(
            f"共发现 {len(missing)} 个缺失文件，暂不生成 OOD 清单。"
        )

    output = AUDIT_DIR / "ood_test.txt"
    content = "\n".join(sorted(ood_paths)) + "\n"

    if output.exists():
        existing = read_list(output)
        require(
            len(existing) == 890
            and {path_key(p) for p in existing} == ood_set,
            "已有 ood_test.txt 与固定划分不一致，请先检查；未覆盖。"
        )
        print("已有 OOD 清单与固定划分一致，无需覆盖。")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as f:
            f.write(content)

    print(f"Train 样本数：{len(train_paths)}")
    print(f"IID 样本数：{len(iid_paths)}")
    print(f"OOD 样本数：{len(ood_paths)}")
    print(f"OOD 有样本地区数：{len({r['region'] for r in ood_rows})}")
    print("三份清单无重复、无交叉，完整覆盖 metadata。")
    print("OOD 影像和标签文件均存在。")
    print(f"清单位置：{output}")


if __name__ == "__main__":
    main()