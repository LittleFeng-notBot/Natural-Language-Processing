"""Task 3: Fine-tune bert-base-uncased for NYT classification.

Config required by the assignment:
  - model: google-bert/bert-base-uncased
  - max_length = 64
  - epochs = 3
"""
import os
import sys

# Probe huggingface.co BEFORE importing transformers; if unreachable (common
# in some regions), switch to the hf-mirror.com mirror. HF_ENDPOINT is read
# at import time by huggingface_hub, so this must happen before the imports.
import socket


def _hf_reachable(host="huggingface.co", timeout=5):
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


if not _hf_reachable():
    print("huggingface.co unreachable -> using mirror https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import RESULTS_DIR, SEED, compute_metrics, load_split, save_metrics, set_seed

# Prefer a local copy of the model (data/bert-base-uncased) so training does
# not depend on network access; fall back to the HuggingFace hub ID.
_LOCAL_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "bert-base-uncased")
if os.path.exists(os.path.join(_LOCAL_MODEL, "model.safetensors")):
    MODEL_NAME = _LOCAL_MODEL
    print(f"Using local model: {MODEL_NAME}")
else:
    MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def get_model(num_labels):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


class NytDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor([label2id[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


def get_dataloader(texts, labels, tokenizer, label2id, shuffle):
    ds = NytDataset(texts, labels, tokenizer, label2id)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=g)


def main():
    set_seed()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label2id = {l: i for i, l in enumerate(["business", "politics", "sports"])}
    id2label = {i: l for l, i in label2id.items()}

    train_texts, y_train = load_split("train")
    val_texts, y_val = load_split("val")
    test_texts, y_test = load_split("test")

    tokenizer = get_tokenizer()
    model = get_model(num_labels=len(label2id)).to(device)

    train_loader = get_dataloader(train_texts, y_train, tokenizer, label2id, shuffle=True)
    val_loader = get_dataloader(val_texts, y_val, tokenizer, label2id, shuffle=False)
    test_loader = get_dataloader(test_texts, y_test, tokenizer, label2id, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={out.loss.item():.4f}")
        print(f"epoch {epoch} avg train loss = {total_loss / len(train_loader):.4f}")

        val_metrics = predict(model, val_loader, id2label, device)
        print(f"epoch {epoch} val: acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    metrics = predict(model, test_loader, id2label, device)
    print(f"TEST: Accuracy = {metrics['accuracy']:.4f}  Macro-F1 = {metrics['macro_f1']:.4f}")
    save_metrics("Task3", "BERT-base-uncased (fine-tuned)", metrics,
                 extra={"max_length": MAX_LENGTH, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR})


@torch.no_grad()
def predict(model, loader, id2label, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        trues.extend(batch["labels"].cpu().tolist())
    y_pred = [id2label[p] for p in preds]
    y_true = [id2label[t] for t in trues]
    return compute_metrics(y_true, y_pred)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
