"""LoRA fine-tuning of a small transformer on the clause topic dataset (CPU only).

Needs the model extras (pip install -e ".[models]") and downloads distilbert-base-uncased on first use.
Writes examples/output/lora_results.json. Run: python examples/lora_finetune.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reg_rag.classification.baseline import accuracy, confusion_matrix, macro_f1  # noqa: E402
from reg_rag.classification.dataset import (  # noqa: E402
    GROUP_COLUMN, LABEL_COLUMN, LABELS, SPLIT_COLUMN, TEXT_COLUMN, build_dataset, group_split,
)
from reg_rag.retriever import load_passages  # noqa: E402

# config
CORPUS_DIR = ROOT / "data" / "corpus"
OUTPUT_PATH = ROOT / "examples" / "output" / "lora_results.json"
MODEL_NAME = "distilbert-base-uncased"
SPLIT_SEED = 0
TRAIN_SEEDS = (0, 1, 2)
EPOCHS = 8
BATCH_SIZE = 16
MAX_LEN = 192
LEARNING_RATE = 5e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_lin", "v_lin"]
NUM_THREADS = 16
DEVICE = "cpu"  # the GPU is deliberately never used
# end config


def encode(tok, texts: list[str]) -> dict:
    return tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts: list[str]) -> list[int]:
    model.eval()
    preds = []
    for i in range(0, len(texts), 32):
        batch = encode(tok, texts[i:i + 32])
        preds += model(**batch).logits.argmax(-1).tolist()
    return preds


def run_seed(seed: int, df, tok) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    label_to_id = {lab: i for i, lab in enumerate(LABELS)}
    train, val, test = (df[df[SPLIT_COLUMN] == s] for s in ("train", "val", "test"))
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)
    model = get_peft_model(base, cfg).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    y_train = np.array([label_to_id[x] for x in train[LABEL_COLUMN]])
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(float)
    weights = torch.tensor(len(y_train) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(train) / BATCH_SIZE))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    texts = train[TEXT_COLUMN].tolist()
    rng = np.random.default_rng(seed)
    best = {"val_macro_f1": -1.0}
    t0 = time.time()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(texts))
        losses = []
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            batch = encode(tok, [texts[j] for j in idx])
            out = model(**batch)
            loss = loss_fn(out.logits, torch.tensor(y_train[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        val_pred = [LABELS[i] for i in predict(model, tok, val[TEXT_COLUMN].tolist())]
        v_f1 = macro_f1(val[LABEL_COLUMN].tolist(), val_pred, LABELS)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": v_f1,
                        "val_accuracy": accuracy(val[LABEL_COLUMN].tolist(), val_pred)})
        print(f"seed {seed} epoch {epoch} loss {np.mean(losses):.3f} val macro-F1 {v_f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if v_f1 > best["val_macro_f1"]:
            test_pred = [LABELS[i] for i in predict(model, tok, test[TEXT_COLUMN].tolist())]
            best = {"epoch": epoch, "val_macro_f1": v_f1, "test_pred": test_pred}
    y_test = test[LABEL_COLUMN].tolist()
    return {"seed": seed, "best_epoch": best["epoch"], "val_macro_f1": best["val_macro_f1"], "trainable_params": trainable,
            "total_params": total, "train_seconds": round(time.time() - t0, 1), "history": history,
            "test_accuracy": accuracy(y_test, best["test_pred"]), "test_macro_f1": macro_f1(y_test, best["test_pred"], LABELS),
            "test_pred": best["test_pred"], "test_true": y_test,
            "confusion": confusion_matrix(y_test, best["test_pred"], LABELS).tolist()}


def main() -> None:
    torch.set_num_threads(NUM_THREADS)
    df = group_split(build_dataset(load_passages(CORPUS_DIR)), seed=SPLIT_SEED)
    assert (df.groupby(GROUP_COLUMN)[SPLIT_COLUMN].nunique() == 1).all(), "section leaked across splits"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = {"model": MODEL_NAME, "split_seed": SPLIT_SEED, "labels": LABELS, "epochs": EPOCHS, "lr": LEARNING_RATE,
               "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
               "n": {s: int((df[SPLIT_COLUMN] == s).sum()) for s in ("train", "val", "test")}, "runs": []}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for seed in TRAIN_SEEDS:
        results["runs"].append(run_seed(seed, df, tok))
        OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")  # saved per seed so a crash loses nothing
    print("done")


if __name__ == "__main__":
    main()
