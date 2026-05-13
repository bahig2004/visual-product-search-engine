from pathlib import Path
import csv
import shutil

BASE = Path(r"C:\clothes_ir_dataset")
IN_FILE = BASE / "processed" / "metadata" / "final_products.csv"
OUT_IMAGES = BASE / "processed" / "images"

def safe_name(name: str) -> str:
    bad = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip()

def main():
    copied = 0
    skipped_missing = 0

    with IN_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            src = Path(row["file_path"])
            category = row["category"].strip().lower()
            image_id = row["image_id"].strip()
            item_id = safe_name(row["item_id"].strip())

            category_folder = OUT_IMAGES / category
            category_folder.mkdir(parents=True, exist_ok=True)

            ext = src.suffix.lower() if src.suffix else ".jpg"
            dst = category_folder / f"{image_id}_{item_id}{ext}"

            if not src.exists():
                skipped_missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

    print("Done.")
    print(f"Copied images: {copied}")
    print(f"Missing skipped: {skipped_missing}")
    print(f"Saved into: {OUT_IMAGES}")

if __name__ == "__main__":
    main()