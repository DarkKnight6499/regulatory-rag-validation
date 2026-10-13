"""Clause topic classification: label quality, keyword baseline, TF-IDF logistic regression and the LoRA results.

Run examples/lora_finetune.py first to include the LoRA row. Needs scikit-learn (pip install -e ".[models]").
Writes examples/output/classifier_eval.json and classifier_eval.md and data/classification/clauses.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reg_rag.classification.baseline import KeywordClassifier, accuracy, confusion_matrix, macro_f1  # noqa: E402
from reg_rag.classification.dataset import (  # noqa: E402
    GROUP_COLUMN, ID_COLUMN, LABEL_COLUMN, LABELS, SPLIT_COLUMN, TEXT_COLUMN, build_dataset, group_split,
)
from reg_rag.metrics import bootstrap_ci  # noqa: E402
from reg_rag.retriever import load_passages  # noqa: E402

# config
CORPUS_DIR = ROOT / "data" / "corpus"
CLAUSES_PATH = ROOT / "data" / "classification" / "clauses.csv"
REVIEW_PATH = ROOT / "data" / "classification" / "hand_review.csv"
LORA_PATH = ROOT / "examples" / "output" / "lora_results.json"
OUTPUT_DIR = ROOT / "examples" / "output"
REVIEW_LABEL_COLUMN = "reviewer_label"
PRIMARY_SPLIT_SEED = 0
REPEAT_SPLIT_SEEDS = (0, 1, 2, 3, 4)
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
N_BOOT = 2000
# end config


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def fit_tfidf_lr(train: pd.DataFrame, val: pd.DataFrame) -> tuple[TfidfVectorizer, LogisticRegression, float]:
    """Choose C on the validation split (a different set of sections than train and test)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)
    xtr = vec.fit_transform(train[TEXT_COLUMN])
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        clf.fit(xtr, train[LABEL_COLUMN])
        f1 = macro_f1(val[LABEL_COLUMN].tolist(), clf.predict(vec.transform(val[TEXT_COLUMN])).tolist(), LABELS)
        if best is None or f1 > best[0]:
            best = (f1, c, clf)
    return vec, best[2], best[1]


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    idx = np.arange(len(y_true))
    yt, yp = np.array(y_true), np.array(y_pred)
    acc = bootstrap_ci((yt == yp).astype(float), N_BOOT, seed=0)
    f1s = []
    rng = np.random.default_rng(0)
    for _ in range(N_BOOT):
        s = rng.choice(idx, size=idx.size, replace=True)
        f1s.append(macro_f1(yt[s].tolist(), yp[s].tolist(), LABELS))
    f1 = macro_f1(y_true, y_pred, LABELS)
    return {"accuracy": acc, "macro_f1": (f1, float(np.quantile(f1s, 0.025)), float(np.quantile(f1s, 0.975))),
            "confusion": confusion_matrix(y_true, y_pred, LABELS).tolist()}


def fmt(ci) -> str:
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


def main() -> None:
    df = build_dataset(load_passages(CORPUS_DIR))
    res: dict = {"n_clauses": len(df), "label_counts": df[LABEL_COLUMN].value_counts().to_dict(), "labels": LABELS}

    review = pd.read_csv(REVIEW_PATH)
    merged = review.merge(df[[ID_COLUMN, LABEL_COLUMN]], on=ID_COLUMN, how="left")
    assert merged[LABEL_COLUMN].notna().all()
    rev, heu = merged[REVIEW_LABEL_COLUMN].tolist(), merged[LABEL_COLUMN].tolist()
    n_none = sum(r == "none" for r in rev)
    dis = merged[merged[REVIEW_LABEL_COLUMN] != merged[LABEL_COLUMN]]
    res["label_quality"] = {
        "n_reviewed": len(merged), "agreement": float(np.mean([a == b for a, b in zip(rev, heu)])),
        "cohen_kappa": cohen_kappa(rev, heu), "reviewer_none": n_none,
        "agreement_excluding_none": float(np.mean([a == b for a, b in zip(rev, heu) if a != "none"])),
        "disagreements": {"count": len(dis), "by_heuristic_label": dis[LABEL_COLUMN].value_counts().to_dict(),
                          "by_reviewer_label": dis[REVIEW_LABEL_COLUMN].value_counts().to_dict()}}

    split = group_split(df, seed=PRIMARY_SPLIT_SEED)
    assert (split.groupby(GROUP_COLUMN)[SPLIT_COLUMN].nunique() == 1).all()
    CLAUSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(CLAUSES_PATH, index=False)
    res["split_sizes"] = {s: int((split[SPLIT_COLUMN] == s).sum()) for s in ("train", "val", "test")}
    res["split_label_counts"] = {s: split[split[SPLIT_COLUMN] == s][LABEL_COLUMN].value_counts().to_dict() for s in ("train", "val", "test")}
    train, val, test = (split[split[SPLIT_COLUMN] == s] for s in ("train", "val", "test"))
    y_test = test[LABEL_COLUMN].tolist()

    majority = train[LABEL_COLUMN].mode()[0]
    res["majority_class"] = {"label": majority, **evaluate(y_test, [majority] * len(y_test))}
    res["keyword_baseline"] = evaluate(y_test, KeywordClassifier().predict(test[TEXT_COLUMN].tolist()))
    vec, clf, best_c = fit_tfidf_lr(train, val)
    res["tfidf_lr"] = evaluate(y_test, clf.predict(vec.transform(test[TEXT_COLUMN])).tolist()) | {"C": best_c}

    rep = {"keyword_baseline": [], "tfidf_lr": []}
    for seed in REPEAT_SPLIT_SEEDS:
        sp = group_split(df, seed=seed)
        tr, va, te = (sp[sp[SPLIT_COLUMN] == s] for s in ("train", "val", "test"))
        yt = te[LABEL_COLUMN].tolist()
        yk = KeywordClassifier().predict(te[TEXT_COLUMN].tolist())
        v, c, _ = fit_tfidf_lr(tr, va)
        yl = c.predict(v.transform(te[TEXT_COLUMN])).tolist()
        rep["keyword_baseline"].append((accuracy(yt, yk), macro_f1(yt, yk, LABELS)))
        rep["tfidf_lr"].append((accuracy(yt, yl), macro_f1(yt, yl, LABELS)))
    res["repeated_splits"] = {k: {"seeds": list(REPEAT_SPLIT_SEEDS), "accuracy_mean": float(np.mean([a for a, _ in v])),
                                  "accuracy_sd": float(np.std([a for a, _ in v], ddof=1)), "macro_f1_mean": float(np.mean([f for _, f in v])),
                                  "macro_f1_sd": float(np.std([f for _, f in v], ddof=1))} for k, v in rep.items()}

    if LORA_PATH.exists():
        lora = json.loads(LORA_PATH.read_text(encoding="utf-8"))
        runs = lora["runs"]
        res["lora"] = {"model": lora["model"], "n_seeds": len(runs), "trainable_params": runs[0]["trainable_params"],
                       "total_params": runs[0]["total_params"], "best_epochs": [r["best_epoch"] for r in runs],
                       "test_accuracy": [r["test_accuracy"] for r in runs], "test_macro_f1": [r["test_macro_f1"] for r in runs],
                       "train_seconds": [r["train_seconds"] for r in runs],
                       "accuracy_mean": float(np.mean([r["test_accuracy"] for r in runs])),
                       "macro_f1_mean": float(np.mean([r["test_macro_f1"] for r in runs])),
                       "seed0": evaluate(runs[0]["test_true"], runs[0]["test_pred"]), "lora_config": lora["lora"], "lr": lora["lr"],
                       "epochs": lora["epochs"]}
        assert runs[0]["test_true"] == y_test, "LoRA test split differs from the evaluation split"
    (OUTPUT_DIR / "classifier_eval.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    write_markdown(res)
    print("wrote", OUTPUT_DIR)


def write_markdown(res: dict) -> None:
    q = res["label_quality"]
    L = ["# Clause topic classification results", "",
         f"{res['n_clauses']} clauses, labels: {res['label_counts']}.", "",
         "## Label quality (hand review of 100 clauses)", "",
         f"Agreement {q['agreement']:.2f} (Cohen kappa {q['cohen_kappa']:.3f}). The reviewer used a `none` label for {q['reviewer_none']} clauses "
         f"that fit no topic (definitions); agreement excluding those is {q['agreement_excluding_none']:.2f}. "
         f"Disagreements: {q['disagreements']['count']}, by heuristic label {q['disagreements']['by_heuristic_label']}.", "",
         "## Split", "", f"Sizes {res['split_sizes']}. Label counts per split: {res['split_label_counts']}.", "",
         "## Test results (primary split, test set only)", "",
         "| Model | Accuracy (95% CI) | Macro F1 (95% CI) |", "|---|---|---|"]
    L.append(f"| Majority class ({res['majority_class']['label']}) | {fmt(res['majority_class']['accuracy'])} | {fmt(res['majority_class']['macro_f1'])} |")
    L.append(f"| Keyword baseline (zero-shot) | {fmt(res['keyword_baseline']['accuracy'])} | {fmt(res['keyword_baseline']['macro_f1'])} |")
    L.append(f"| TF-IDF logistic regression (C={res['tfidf_lr']['C']}) | {fmt(res['tfidf_lr']['accuracy'])} | {fmt(res['tfidf_lr']['macro_f1'])} |")
    if "lora" in res:
        lo = res["lora"]
        L.append(f"| LoRA {lo['model']}, seed 0 | {fmt(lo['seed0']['accuracy'])} | {fmt(lo['seed0']['macro_f1'])} |")
        L.append(f"| LoRA mean over {lo['n_seeds']} seeds | {lo['accuracy_mean']:.3f} | {lo['macro_f1_mean']:.3f} |")
    L += ["", "Repeated group splits (mean and sd over split seeds; test sets are small and differ by seed):", "",
          "| Model | Accuracy | Macro F1 |", "|---|---|---|"]
    for k, v in res["repeated_splits"].items():
        L.append(f"| {k} | {v['accuracy_mean']:.3f} (sd {v['accuracy_sd']:.3f}) | {v['macro_f1_mean']:.3f} (sd {v['macro_f1_sd']:.3f}) |")
    if "lora" in res:
        lo = res["lora"]
        L += ["", f"LoRA per-seed test accuracy {['%.3f' % x for x in lo['test_accuracy']]}, macro F1 {['%.3f' % x for x in lo['test_macro_f1']]}, "
              f"best epochs {lo['best_epochs']}, trainable parameters {lo['trainable_params']:,} of {lo['total_params']:,}, "
              f"training seconds {lo['train_seconds']}. Config: {lo['lora_config']}, learning rate {lo['lr']}, {lo['epochs']} epochs."]
    labs = res["labels"]
    for name, key in (("Keyword baseline", "keyword_baseline"), ("TF-IDF logistic regression", "tfidf_lr")) + ((("LoRA seed 0", None),) if "lora" in res else ()):
        m = res[key]["confusion"] if key else res["lora"]["seed0"]["confusion"]
        L += ["", f"### Confusion matrix: {name} (rows true, columns predicted)", "", "| true \\ pred | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for lab, row in zip(labs, m):
            L.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    L.append("")
    (OUTPUT_DIR / "classifier_eval.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
