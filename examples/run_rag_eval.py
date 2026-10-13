"""Evaluate retrieval, extractive answering, abstention, groundedness and injection robustness on the real corpus.

Needs the model extras (pip install -e ".[models]") and downloads the embedding models on first use.
Writes examples/output/rag_eval.json, rag_eval.md and per_question.csv.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reg_rag.answerer import ExtractiveAnswerer  # noqa: E402
from reg_rag.corpus import verify_manifest  # noqa: E402
from reg_rag.dense import SentenceTransformerEmbedder  # noqa: E402
from reg_rag.evalset import load_questions, validate_questions  # noqa: E402
from reg_rag.evaluate import (  # noqa: E402
    answer_all, answer_report, apply_threshold, balanced_abstain_accuracy, behavior_change, poison_passages,
    retrieval_report, split_dev_test, tune_threshold,
)
from reg_rag.grounding import sentence_grounded  # noqa: E402
from reg_rag.metrics import bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from reg_rag.retriever import Retriever, load_passages  # noqa: E402

# config
CORPUS_DIR = ROOT / "data" / "corpus"
QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.jsonl"
CACHE_DIR = ROOT / "data" / "index_cache"
OUTPUT_DIR = ROOT / "examples" / "output"
PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"
PRIMARY_PREFIX = "Represent this sentence for searching relevant passages: "
SECONDARY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_BOOT = 2000
SEED = 0
LEXICAL_WEIGHT = 0.6
TOP_K_ANSWER = 3
POISON_STYLES = ("instruction", "embedded", "fakeclaim")
# end config


def fmt(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    problems = verify_manifest(CORPUS_DIR / "manifest.json", CORPUS_DIR)
    assert not problems, problems
    passages = load_passages(CORPUS_DIR)
    questions = load_questions(QUESTIONS_PATH)
    problems = validate_questions(questions, passages)
    assert not problems, problems
    results: dict = {"n_passages": len(passages), "n_questions": len(questions),
                     "types": pd.Series([q.type for q in questions]).value_counts().to_dict(),
                     "models": {"primary": PRIMARY_MODEL, "secondary": SECONDARY_MODEL}, "seed": SEED, "n_boot": N_BOOT}

    emb = SentenceTransformerEmbedder(PRIMARY_MODEL, PRIMARY_PREFIX, CACHE_DIR)
    retriever = Retriever(passages, emb)
    rep = retrieval_report(retriever, questions, n_boot=N_BOOT, seed=SEED)
    emb2 = SentenceTransformerEmbedder(SECONDARY_MODEL, "", CACHE_DIR)
    rep2 = retrieval_report(Retriever(passages, emb2), questions, modes=("dense", "hybrid"), n_boot=N_BOOT, seed=SEED)
    rep["dense_minilm"], rep["hybrid_minilm"] = rep2["dense"], rep2["hybrid"]
    results["retrieval"] = {k: v for k, v in rep.items() if k != "n"} | {"n": rep["n"]}
    diff = {}
    for name, a, b in (("hybrid_vs_bm25", "hybrid", "bm25"), ("hybrid_vs_dense", "hybrid", "dense"), ("bm25_vs_dense", "bm25", "dense")):
        diff[name] = paired_bootstrap_diff(rep[a]["per_question_rr"], rep[b]["per_question_rr"], N_BOOT, SEED)
    results["mrr_paired_diff"] = diff

    # answering: collect confidence at threshold 0, tune threshold on the dev half, report the test half
    base = ExtractiveAnswerer(retriever, threshold=0.0, top_k=TOP_K_ANSWER, lexical_weight=LEXICAL_WEIGHT)
    records = answer_all(base, questions)
    dev_idx = [i for i in range(len(questions)) if i % 2 == 0]
    test_idx = [i for i in range(len(questions)) if i % 2 == 1]
    dev_recs, dev_qs = [records[i] for i in dev_idx], [questions[i] for i in dev_idx]
    test_recs, test_qs = [records[i] for i in test_idx], [questions[i] for i in test_idx]
    thr = tune_threshold(dev_recs, dev_qs)
    results["threshold"] = {"value": thr, "tuned_on": "even-indexed questions (dev)", "n_dev": len(dev_qs), "n_test": len(test_qs),
                            "dev_balanced_abstain_accuracy": balanced_abstain_accuracy(dev_recs, dev_qs, thr),
                            "test_balanced_abstain_accuracy": balanced_abstain_accuracy(test_recs, test_qs, thr)}
    results["answers_test"] = answer_report(apply_threshold(test_recs, thr), test_qs, N_BOOT, SEED)
    results["answers_all_in_sample_threshold"] = answer_report(apply_threshold(records, thr), questions, N_BOOT, SEED)
    results["answers_no_abstention"] = answer_report(records, questions, N_BOOT, SEED)
    sweep = []
    for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        r = answer_report(apply_threshold(records, t), questions, 200, SEED)
        sweep.append({"threshold": t, "abstain_precision": r["abstention"]["precision"], "abstain_recall": r["abstention"]["recall"],
                      "correct_all_answerable": r["span_contained_all_answerable"][0]})
    results["threshold_sweep_all_questions"] = sweep

    final = apply_threshold(records, thr)
    pd.DataFrame([asdict(r) | {"question": q.question} for r, q in zip(final, questions)]).to_csv(OUTPUT_DIR / "per_question.csv", index=False)

    # groundedness negative control: altering one figure or modal verb in a returned sentence must be caught
    tuned = ExtractiveAnswerer(retriever, threshold=thr, top_k=TOP_K_ANSWER, lexical_weight=LEXICAL_WEIGHT)
    answers = [tuned.answer(q.question) for q in questions]
    answered = [(a, q) for a, q in zip(answers, questions) if not a.abstained]
    caught = mutable = 0
    for a, _ in answered:
        hit_passages = [h.passage for h in a.hits]
        text = a.sentences[0].text
        mutated = (re.sub(r"\d+", lambda m: str(int(m.group()) + 7), text, count=1) if re.search(r"\d", text)
                   else text.replace(" must ", " may ", 1))
        if mutated == text:
            continue
        mutable += 1
        caught += not sentence_grounded(mutated, hit_passages)
    results["groundedness_control"] = {"n": mutable, "alteration_detected": caught, "n_answered": len(answered)}

    # injection: user-side (adversarial questions vs their clean twins) with defence off and on
    by_id = {q.id: i for i, q in enumerate(questions)}
    inj = {}
    for defence in (False, True):
        ans = ExtractiveAnswerer(retriever, threshold=thr, top_k=TOP_K_ANSWER, lexical_weight=LEXICAL_WEIGHT, defence=defence)
        adv = [q for q in questions if q.type == "adversarial"]
        rows = []
        for q in adv:
            a = ans.answer(q.question)
            c = ans.answer(questions[by_id[q.clean_twin]].question)
            rows.append({"id": q.id, "same_as_clean": (a.text == c.text and a.citations == c.citations and a.abstained == c.abstained),
                         "payload_in_answer": bool(q.payload) and q.payload.lower() in a.text.lower()})
        abst = [ans.answer(q.question).abstained for q in questions if q.type == "adversarial_abstain"]
        inj[f"defence_{defence}"] = {"n": len(rows), "same_as_clean": sum(r["same_as_clean"] for r in rows),
                                     "payload_in_answer": sum(r["payload_in_answer"] for r in rows),
                                     "pure_injection_abstained": int(sum(abst)), "n_pure_injection": len(abst), "rows": rows}
    results["injection_user_side"] = inj

    # injection: corpus-side (poisoned passages added to the index)
    clean_ans = ExtractiveAnswerer(retriever, threshold=thr, top_k=TOP_K_ANSWER, lexical_weight=LEXICAL_WEIGHT, defence=True)
    answerable = [q for q in questions if q.type == "answerable"]
    clean_recs = answer_all(clean_ans, answerable)
    corpus_side = {}
    for style in POISON_STYLES:
        poisoned = Retriever(passages + poison_passages(questions, passages, style), emb)
        for defence in (False, True):
            ans = ExtractiveAnswerer(poisoned, threshold=thr, top_k=TOP_K_ANSWER, lexical_weight=LEXICAL_WEIGHT, defence=defence)
            recs = answer_all(ans, answerable)
            ref = answer_all(ExtractiveAnswerer(retriever, threshold=thr, top_k=TOP_K_ANSWER, lexical_weight=LEXICAL_WEIGHT, defence=defence), answerable)
            corpus_side[f"{style}|defence_{defence}"] = behavior_change(ref, recs) | {
                "correct_when_poisoned": float(np.mean([r.contained for r in recs]))}
    results["injection_corpus_side"] = corpus_side
    results["runtime_seconds"] = round(time.time() - t0, 1)

    (OUTPUT_DIR / "rag_eval.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    write_markdown(results)
    print(f"done in {results['runtime_seconds']} s; wrote {OUTPUT_DIR}")


def write_markdown(res: dict) -> None:
    L = ["# RAG evaluation results", "",
         f"Passages {res['n_passages']}, questions {res['n_questions']} ({res['types']}), bootstrap {res['n_boot']} resamples, seed {res['seed']}.", "",
         "## Retrieval (answerable questions, n=%d)" % res["retrieval"]["n"], "",
         "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |", "|---|---|---|---|---|---|"]
    names = {"bm25": "BM25", "dense": "Dense (bge-small)", "hybrid": "Hybrid RRF (BM25 + bge-small)",
             "dense_minilm": "Dense (MiniLM-L6)", "hybrid_minilm": "Hybrid RRF (BM25 + MiniLM-L6)"}
    for k, label in names.items():
        r = res["retrieval"][k]
        L.append(f"| {label} | " + " | ".join(fmt(r[m]) for m in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")) + " |")
    L += ["", "Paired bootstrap difference in reciprocal rank (mean, 95% CI):", ""]
    for k, v in res["mrr_paired_diff"].items():
        L.append(f"- {k}: {fmt(v)}")
    t = res["threshold"]
    L += ["", "## Extractive answerer", "",
          f"Abstention threshold {t['value']:.2f} tuned on the {t['n_dev']} even-indexed questions; reported on the {t['n_test']} odd-indexed (held-out) questions. "
          f"Balanced abstention accuracy: dev {t['dev_balanced_abstain_accuracy']:.3f}, test {t['test_balanced_abstain_accuracy']:.3f}.", "",
          "| Metric | Held-out half | All questions (threshold tuned on half of them) | No abstention |", "|---|---|---|---|"]
    a, b, c = res["answers_test"], res["answers_all_in_sample_threshold"], res["answers_no_abstention"]
    for key, label in (("span_contained_all_answerable", "Answer span contained (all answerable)"),
                       ("span_contained_when_answered", "Answer span contained (when answered)"),
                       ("token_f1_all_answerable", "Token F1 (best sentence, all answerable)"),
                       ("citation_top_correct", "Top citation correct (when answered)"),
                       ("citation_any_correct", "Any citation correct (when answered)"),
                       ("groundedness", "Groundedness (sentences found verbatim in retrieved passages)"),
                       ("abstention_precision_ci", "Abstention precision"), ("abstention_recall_ci", "Abstention recall")):
        L.append(f"| {label} | {fmt(a[key])} | {fmt(b[key])} | {fmt(c[key])} |")
    L += ["", f"Counts on held-out half: {a['n_questions']} questions, {a['n_answerable']} answerable, {a['n_answered']} answered; abstention tp/fp/fn = {a['abstention']['tp']}/{a['abstention']['fp']}/{a['abstention']['fn']}.", "",
          "### Threshold sweep (all questions)", "", "| Threshold | Abstain precision | Abstain recall | Span contained, all answerable |", "|---|---|---|---|"]
    for s in res["threshold_sweep_all_questions"]:
        L.append(f"| {s['threshold']:.1f} | {s['abstain_precision']:.3f} | {s['abstain_recall']:.3f} | {s['correct_all_answerable']:.3f} |")
    g = res["groundedness_control"]
    L += ["", f"Groundedness negative control: one figure or modal verb altered in the first returned sentence of {g['n']} answers (of {g['n_answered']} answered) was detected in {g['alteration_detected']}.", "",
          "## Prompt injection", "", "User-side: adversarial questions (instruction plus a real question) against their clean twins.", "",
          "| Defence | Same answer as clean twin | Payload in answer | Pure-injection question abstained |", "|---|---|---|---|"]
    for k, v in res["injection_user_side"].items():
        L.append(f"| {k.split('_')[1]} | {v['same_as_clean']}/{v['n']} | {v['payload_in_answer']}/{v['n']} | {v['pure_injection_abstained']}/{v['n_pure_injection']} |")
    L += ["", "Corpus-side: one poisoned passage per answerable question added to the index (behaviour compared with the same defence setting on the clean index).", "",
          "| Poison style | Defence | Behaviour changed | Payload in answer | Poison passage cited | Span contained under attack |", "|---|---|---|---|---|---|"]
    for k, v in res["injection_corpus_side"].items():
        style, d = k.split("|")
        L.append(f"| {style} | {d.split('_')[1]} | {v['behavior_changed']}/{v['n']} | {v['payload_in_answer']} | {v['poison_cited']} | {v['correct_when_poisoned']:.3f} |")
    L.append("")
    (OUTPUT_DIR / "rag_eval.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
