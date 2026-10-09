"""Shuffle the NYT dataset (seed=42) and split it 80/10/10 into
train / validation / test CSV files under data/."""
import csv
import os
import random

from utils import DATA_DIR, SEED


def main():
    src = os.path.join(DATA_DIR, "nyt.csv")
    with open(src, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} rows from {src}")

    random.seed(SEED)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }
    for name, split in splits.items():
        out = os.path.join(DATA_DIR, f"nyt_{name}.csv")
        with open(out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(split)
        print(f"Wrote {len(split)} rows -> {out}")


if __name__ == "__main__":
    main()
