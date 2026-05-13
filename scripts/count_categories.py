from pathlib import Path
import csv
from collections import Counter

BASE = Path(r"C:\clothes_ir_dataset")
IN_FILE = BASE / "processed" / "metadata" / "final_products.csv"
OUT_FILE = BASE / "processed" / "metadata" / "category_counts.csv"

def main():
    counter = Counter()
    total_rows = 0

    with IN_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"].strip().lower()
            counter[category] += 1
            total_rows += 1

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count"])
        for category, count in sorted(counter.items()):
            writer.writerow([category, count])
        writer.writerow(["TOTAL", total_rows])

    print("Done.")
    print(f"Total rows: {total_rows}")
    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()