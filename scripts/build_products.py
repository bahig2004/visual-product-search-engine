from pathlib import Path
import csv

BASE = Path(r"C:\clothes_ir_dataset")
RAW = BASE / "raw" / "deepfashion"

# Your real extracted image path is:
# C:\clothes_ir_dataset\raw\deepfashion\Img\img\img\...
IMG_BASE = RAW / "Img" / "img" / "img"

CATEGORY_CLOTH_FILE = RAW / "Anno_coarse" / "list_category_cloth.txt"
CATEGORY_IMG_FILE = RAW / "Anno_coarse" / "list_category_img.txt"
EVAL_FILE = RAW / "Eval" / "list_eval_partition.txt"
SELECTED_LABELS_FILE = BASE / "processed" / "metadata" / "selected_category_labels.txt"

OUT_FILE = BASE / "processed" / "metadata" / "products.csv"


def load_selected_labels(path: Path) -> dict[int, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split("->")
            mapping[int(left.strip())] = right.strip()
    return mapping


def load_category_names(path: Path) -> dict[int, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # line 1 = count, line 2 = header
    data_lines = lines[2:]

    for label, line in enumerate(data_lines, start=1):
        parts = line.split()
        category_name = " ".join(parts[:-1])
        mapping[label] = category_name

    return mapping


def load_image_labels(path: Path) -> dict[str, int]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # line 1 = count, line 2 = header
    data_lines = lines[2:]

    for line in data_lines:
        image_name, label = line.split()
        mapping[image_name] = int(label)

    return mapping


def load_eval_splits(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # line 1 = count, line 2 = header
    data_lines = lines[2:]

    for line in data_lines:
        image_name, split_name = line.split()
        mapping[image_name] = split_name

    return mapping


def main():
    selected_label_to_project_category = load_selected_labels(SELECTED_LABELS_FILE)
    label_to_deepfashion_category = load_category_names(CATEGORY_CLOTH_FILE)
    image_to_label = load_image_labels(CATEGORY_IMG_FILE)
    image_to_split = load_eval_splits(EVAL_FILE)

    rows = []
    image_counter = 1

    for relative_image_path, label in image_to_label.items():
        if label not in selected_label_to_project_category:
            continue

        project_category = selected_label_to_project_category[label]
        deepfashion_subcategory = label_to_deepfashion_category.get(label, "")
        split_name = image_to_split.get(relative_image_path, "")

        # list_category_img.txt paths start with img/...
        # but your local folder is already ...\Img\img\img
        # so remove the first "img/" before joining
        relative_without_first_img = relative_image_path.replace("img/", "", 1)
        full_image_path = IMG_BASE / relative_without_first_img

        item_id = Path(relative_image_path).parent.name

        if full_image_path.exists():
            is_valid = "pending"
        else:
            is_valid = "missing"

        rows.append({
            "image_id": f"img_{image_counter:06d}",
            "item_id": item_id,
            "file_path": str(full_image_path),
            "category": project_category,
            "subcategory": deepfashion_subcategory,
            "source_type": "deepfashion",
            "width": "",
            "height": "",
            "is_valid": is_valid,
            "is_duplicate": "pending",
            "notes": f"split={split_name}",
        })

        image_counter += 1

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "item_id",
                "file_path",
                "category",
                "subcategory",
                "source_type",
                "width",
                "height",
                "is_valid",
                "is_duplicate",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Rows written: {len(rows)}")
    print(f"Saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()