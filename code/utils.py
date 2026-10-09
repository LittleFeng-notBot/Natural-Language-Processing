"""Shared utilities for HW-1: text classification on the NYT dataset."""
import csv
import os
import random
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

SEED = 42
LABELS = ["business", "politics", "sports"]

_token_cache = {}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def tokenize(text):
    """Simple tokenizer: lowercase alphanumeric sequences.

    Equivalent in spirit to nltk.word_tokenize for this lowercase news text,
    but without any external model downloads.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_split(name):
    """Load one of nyt_train / nyt_val / nyt_test CSV files."""
    path = os.path.join(DATA_DIR, f"nyt_{name}.csv")
    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def compute_metrics(y_true, y_pred):
    """Return Accuracy and Macro-F1."""
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_metrics(task, method, metrics, extra=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {"task": task, "method": method, **metrics}
    if extra:
        record.update(extra)
    out = os.path.join(RESULTS_DIR, "results.jsonl")
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(str(record).replace("'", '"') + "\n")
    print(f"[saved] {record}")
