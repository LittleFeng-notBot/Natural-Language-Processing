"""Run every experiment in order: Task 1, Task 2, Task 3."""
import subprocess
import sys
import time

STEPS = [
    ("Task1 Binary BoW", [sys.executable, "task1_bow.py", "binary"]),
    ("Task1 Word Frequency", [sys.executable, "task1_bow.py", "freq"]),
    ("Task2 GloVe", [sys.executable, "task2_word2vec.py", "glove"]),
    ("Task2 Word2Vec on AG", [sys.executable, "task2_word2vec.py", "w2v_ag"]),
    ("Task2 Word2Vec on NYT", [sys.executable, "task2_word2vec.py", "w2v_nyt"]),
    ("Task3 BERT", [sys.executable, "task3_bert.py"]),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, cmd in STEPS:
        if only and only not in name:
            continue
        print(f"\n########## {name} ##########", flush=True)
        t0 = time.time()
        ret = subprocess.call(cmd)
        print(f"########## {name} finished in {time.time() - t0:.0f}s (exit {ret}) ##########")
