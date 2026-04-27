from pathlib import Path
import csv
from PIL import Image

BASE = Path(r"C:\clothes_ir_dataset")
IN_FILE = BASE / "processed" / "metadata" / "products.csv"
OUT_FILE = BASE / "processed" / "metadata" / "products_validated.csv"

def main():
    rows = []

    with IN_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(row["file_path"])

            if not image_path.exists():
                row["is_valid"] = "missing"
                row["width"] = ""
                row["height"] = ""
                row["notes"] = row["notes"] + "; file_missing"
            else:
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                    row["width"] = str(width)
                    row["height"] = str(height)
                    row["is_valid"] = "yes"
                except Exception:
                    row["width"] = ""
                    row["height"] = ""
                    row["is_valid"] = "corrupted"
                    row["notes"] = row["notes"] + "; image_open_failed"

            rows.append(row)

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
    print(f"Validated rows: {len(rows)}")
    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()