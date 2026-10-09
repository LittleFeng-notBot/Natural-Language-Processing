"""Task 1: Bag-of-Words text representations + Logistic Regression.

Methods:
  1. Binary Bag of Words  (x_i = 1 if word i appears in the document)
  2. Word Frequency       (x_i = count of word i in the document)

Vocabulary is built from the NYT *training* split only.
"""
import os
import sys

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from utils import RESULTS_DIR, compute_metrics, load_split, save_metrics, set_seed, tokenize


def build_vocab(train_texts, min_df=1):
    """Map every training-corpus word to an index. min_df drops words that
    appear in fewer than `min_df` documents (min_df=1 keeps the full vocab)."""
    from collections import Counter

    df = Counter()
    for t in train_texts:
        df.update(set(tokenize(t)))
    words = sorted(w for w, c in df.items() if c >= min_df)
    return {w: i for i, w in enumerate(words)}


def vectorize(texts, vocab, binary):
    """Build a CSR matrix (n_docs x |V|) of 0/1 or counts."""
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        counts = {}
        for w in tokenize(t):
            j = vocab.get(w)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            rows.append(i)
            cols.append(j)
            vals.append(1 if binary else c)
    n_docs, n_feats = len(texts), len(vocab)
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(n_docs, n_feats),
    )


def run(name, binary):
    print(f"\n=== Task 1: {name} ===")
    train_texts, y_train = load_split("train")
    test_texts, y_test = load_split("test")

    vocab = build_vocab(train_texts)
    print(f"Vocabulary size |V| = {len(vocab)}")

    X_train = vectorize(train_texts, vocab, binary=binary)
    X_test = vectorize(test_texts, vocab, binary=binary)
    print(f"Feature matrices: train {X_train.shape}, test {X_test.shape}")

    set_seed()
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print(f"Accuracy = {metrics['accuracy']:.4f}  Macro-F1 = {metrics['macro_f1']:.4f}")
    save_metrics("Task1", name, metrics, extra={"vocab_size": len(vocab)})

    # keep the fitted vectorizer artifacts out of memory; nothing to persist
    return metrics


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("binary", "both"):
        run("Binary Bag of Words", binary=True)
    if which in ("freq", "both"):
        run("Word Frequency", binary=False)
