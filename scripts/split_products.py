from pathlib import Path
import csv

BASE = Path(r"C:\clothes_ir_dataset")
IN_FILE = BASE / "processed" / "metadata" / "final_products.csv"
SPLITS_DIR = BASE / "processed" / "splits"

TRAIN_FILE = SPLITS_DIR / "train.csv"
VAL_FILE = SPLITS_DIR / "val.csv"
TEST_FILE = SPLITS_DIR / "test.csv"

def extract_split(notes: str) -> str:
    for part in notes.split(";"):
        part = part.strip().lower()
        if part.startswith("split="):
            return part.split("=", 1)[1].strip()
    return ""

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = []
    val_rows = []
    test_rows = []

    with IN_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            split_name = extract_split(row.get("notes", ""))

            if split_name == "train":
                train_rows.append(row)
            elif split_name == "val":
                val_rows.append(row)
            elif split_name == "test":
                test_rows.append(row)

    for out_file, rows in [
        (TRAIN_FILE, train_rows),
        (VAL_FILE, val_rows),
        (TEST_FILE, test_rows),
    ]:
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("Done.")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Saved to: {SPLITS_DIR}")

if __name__ == "__main__":
    main()