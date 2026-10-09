"""Task 2: Word-Embedding document representations (100-d) + Logistic Regression.

Methods:
  1. Pre-trained GloVe (glove.6B.100d) — document = mean of its word vectors
  2. Word2Vec trained on AG News (gensim, 100-d)
  3. Word2Vec trained on NYT training split (gensim, 100-d)
"""
import csv
import os
import sys
import zipfile

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from utils import DATA_DIR, RESULTS_DIR, compute_metrics, load_split, save_metrics, set_seed, tokenize

GLOVE_DIR = os.path.join(DATA_DIR, "glove")
GLOVE_TXT = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")


def load_glove():
    """Load the 100-d GloVe vectors; unzip glove.6B.zip first if needed."""
    if not os.path.exists(GLOVE_TXT):
        zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
        print("Unzipping glove.6B.zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("glove.6B.100d.txt", GLOVE_DIR)
    embeddings = {}
    with open(GLOVE_TXT, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            embeddings[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    print(f"Loaded {len(embeddings)} GloVe vectors")
    return embeddings


def train_word2vec(corpus_name):
    """Train a 100-d Word2Vec model on AG News or the NYT training split."""
    out_path = os.path.join(DATA_DIR, f"word2vec_{corpus_name}.model")
    if os.path.exists(out_path):
        print(f"Loading existing Word2Vec model: {out_path}")
        return Word2Vec.load(out_path)

    if corpus_name == "ag":
        src = os.path.join(DATA_DIR, "ag.csv")
        with open(src, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            sentences = [tokenize(row["text"]) for row in reader]
    elif corpus_name == "nyt":
        texts, _ = load_split("train")
        sentences = [tokenize(t) for t in texts]
    else:
        raise ValueError(corpus_name)
    print(f"Training Word2Vec on {corpus_name}: {len(sentences)} documents")

    set_seed()
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=0,           # CBOW
        epochs=5,
        seed=42,
    )
    model.save(out_path)
    return model


def doc_vectors(texts, lookup):
    """Mean of the embeddings of all valid words; zero vector for empty docs."""
    dim = 100
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = []
        for w in tokenize(t):
            e = lookup(w)
            if e is not None:
                vecs.append(e)
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return X


def evaluate(name, lookup):
    train_texts, y_train = load_split("train")
    test_texts, y_test = load_split("test")
    X_train = doc_vectors(train_texts, lookup)
    X_test = doc_vectors(test_texts, lookup)

    set_seed()
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    print(f"Accuracy = {metrics['accuracy']:.4f}  Macro-F1 = {metrics['macro_f1']:.4f}")
    save_metrics("Task2", name, metrics)
    return metrics


def run(which):
    if which == "glove":
        emb = load_glove()
        evaluate("Pre-trained GloVe (100d)", lambda w: emb.get(w))
    elif which == "w2v_ag":
        model = train_word2vec("ag")
        evaluate("Word2Vec on AG News (100d)", lambda w: model.wv[w] if w in model.wv else None)
    elif which == "w2v_nyt":
        model = train_word2vec("nyt")
        evaluate("Word2Vec on NYT (100d)", lambda w: model.wv[w] if w in model.wv else None)
    else:
        raise ValueError(which)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("glove", "all"):
        run("glove")
    if which in ("w2v_ag", "all"):
        run("w2v_ag")
    if which in ("w2v_nyt", "all"):
        run("w2v_nyt")
