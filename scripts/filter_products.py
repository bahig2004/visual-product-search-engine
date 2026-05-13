from pathlib import Path
import csv

BASE = Path(r"C:\clothes_ir_dataset")
IN_FILE = BASE / "processed" / "metadata" / "products_validated.csv"
OUT_FILE = BASE / "processed" / "metadata" / "final_products.csv"

MIN_WIDTH = 224
MIN_HEIGHT = 224

def main():
    kept_rows = []
    removed_count = 0
    too_small_count = 0
    invalid_count = 0

    with IN_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_valid = row["is_valid"].strip().lower()

            if is_valid != "yes":
                removed_count += 1
                invalid_count += 1
                continue

            try:
                width = int(row["width"])
                height = int(row["height"])
            except Exception:
                removed_count += 1
                invalid_count += 1
                continue

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                removed_count += 1
                too_small_count += 1
                continue

            kept_rows.append(row)

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
        writer.writerows(kept_rows)

    print("Done.")
    print(f"Kept rows: {len(kept_rows)}")
    print(f"Removed rows: {removed_count}")
    print(f"Too small removed: {too_small_count}")
    print(f"Invalid removed: {invalid_count}")
    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()