# regulatory-rag-validation

A retrieval-augmented question-answering assistant over primary US bank regulatory texts (liquidity, capital and model risk), with an evaluation harness, a clause-classifier comparison (keyword baseline, TF-IDF logistic regression, LoRA fine-tuning) and an SR 26-2 style validation report of the assistant itself.

The assistant is extractive: it returns cited sentences from the corpus and abstains when evidence is weak. A generative step can be plugged in behind an interface, but none is needed and no API key is used anywhere.

Not legal advice. The corpus is a dated snapshot of rules that change.

## Corpus

764 passages from 8 documents, about 470 KB, each file hashed in `data/corpus/manifest.json` (source URL, retrieval date, SHA-256; the eCFR files and FR 2052a were retrieved 2026-10-12, SR 26-2 on 2026-10-13).

| Document | Source |
|---|---|
| 12 CFR 249 (Liquidity Coverage Ratio) | eCFR, version 2026-10-01 |
| 12 CFR 252 subparts D and E (liquidity risk management, stress testing) | eCFR, version 2026-10-01 |
| 12 CFR 217 subparts B and C, section 217.30 (capital) | eCFR, version 2026-10-01 |
| SR 26-2 attachment (interagency model risk management guidance of April 17, 2026, which replaced SR 11-7 and SR 21-8) | federalreserve.gov PDF |
| FR 2052a instructions, general instructions and field definitions (form version dated 2022-04-29, possibly not current) | federalreserve.gov PDF |

Chunking follows legal structure (section, then paragraph tree), and every passage carries a citation string and the paragraph keys it covers, for example `12 CFR 249.10(a)` or `SR 26-2, V (Outcomes Analysis), para 2`.

## Quick start

Python 3.11 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # macOS or Linux: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests need only numpy and pandas: they use a small synthetic corpus, no network and no model downloads, and they also check the committed corpus manifest and evaluation set.

To reproduce the results (CPU only, downloads models from Hugging Face on first use):

```
python -m pip install -e ".[models]"
python scripts/fetch_corpus.py            # optional: refetch the corpus (changes hashes if the rules changed)
python examples/run_rag_eval.py           # downloads bge-small-en-v1.5 and all-MiniLM-L6-v2
python examples/lora_finetune.py          # downloads distilbert-base-uncased, about 20 minutes on a laptop CPU
python examples/run_classifier_eval.py    # reads the LoRA results if present
```

Outputs go to `examples/output/` (`rag_eval.md` and `.json`, `per_question.csv`, `classifier_eval.md` and `.json`, `lora_results.json`). Passage embeddings are cached in `data/index_cache/`.

Use as a library:

```python
from pathlib import Path
from reg_rag.answerer import ExtractiveAnswerer
from reg_rag.dense import SentenceTransformerEmbedder
from reg_rag.retriever import Retriever, load_passages

passages = load_passages(Path("data/corpus"))
embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: ")
answerer = ExtractiveAnswerer(Retriever(passages, embedder), threshold=0.64)
answer = answerer.answer("What is effective challenge?")
print(answer.text, answer.citations)
```

To use a generative model, wrap any `prompt -> text` function in `CallableGenerator` and pass it to `GenerativeAnswerer`. A generation that is not grounded in the retrieved passages, or that looks like an instruction, is replaced by the extractive answer. This hook is tested only with a stub function.

## Evaluation set

66 questions written from the corpus text, each answerable one with gold paragraph keys and an answer span that is checked against the corpus by `validate_questions`: 52 answerable, 8 unanswerable (should abstain), 5 adversarial (a real question plus an injected instruction, each with a clean twin), 1 pure-injection question (should be refused). Intervals are 95% percentile bootstrap, 2,000 resamples, seed 0.

## Results

### Retrieval, 52 answerable questions

| Retriever | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.712 | 0.962 | 0.810 |
| Dense, bge-small | 0.769 | 0.923 | 0.817 |
| Hybrid (RRF), BM25 + bge-small | 0.731 | 0.962 | 0.828 |
| Dense, MiniLM-L6 | 0.673 | 0.885 | 0.775 |
| Hybrid (RRF), BM25 + MiniLM-L6 | 0.769 | 0.981 | 0.854 |

Intervals are wide (for example BM25 MRR 0.810, CI 0.722 to 0.892). Paired MRR difference, hybrid minus BM25: 0.018 (CI -0.057 to 0.092), and hybrid minus dense: 0.010 (CI -0.038 to 0.057), so on this corpus the three retrievers are not distinguishable.

### Answers and abstention

The abstention threshold (0.64) was tuned on the 33 even-indexed questions and reported on the 33 odd-indexed ones.

| Metric | Held-out half | All questions | No abstention |
|---|---|---|---|
| Answer span contained, all answerable | 0.577 | 0.635 | 0.692 |
| Answer span contained, when answered | 0.682 | 0.733 | 0.692 |
| Top citation correct, when answered | 0.773 | 0.822 | 0.750 |
| Groundedness (every sentence verbatim in retrieved passages) | 1.000 | 1.000 | 1.000 |
| Abstention precision | 0.500 | 0.500 | 1.000 |
| Abstention recall | 0.800 | 0.889 | 0.111 |

Balanced abstention accuracy was 0.931 on the dev half and 0.829 on the held-out half. Abstention is the weakest control: confidence for unanswerable questions (0.356 to 0.653) overlaps that of answerable ones (0.407 to 0.959). Changing one figure or modal verb in a returned sentence was caught by the groundedness check in 30 of 30 applicable answers.

### Prompt injection

| Test | Defence off | Defence on |
|---|---|---|
| Instruction in the question: payload appears in answer | 0 of 5 | 0 of 5 |
| Same answer as the question's clean twin | 0 of 5 | 4 of 5 |
| Corpus poisoned, instruction style: behaviour changed | 52 of 52 | 1 of 52 |
| Corpus poisoned, instruction embedded in a copy of the gold passage: behaviour changed (payload in answer) | 29 of 52 (0) | 2 of 52 (0) |
| Corpus poisoned with a false claim and no instruction wording: behaviour changed (payload in answer) | 19 of 52 (8) | 19 of 52 (8) |

The pattern defence stops instruction-style attacks and cannot stop a plausible false statement, which is the main open security finding.

### Clause topic classifier

727 clauses in 7 topics, labelled by document and section (structural rules), split by section so no section spans two splits (train 441, validation 146, test 140). The smallest class, model_validation, has 10 clauses (2 in test), so its scores are close to anecdotes. Hand review of 100 clauses (a single AI reviewer): agreement 0.84, Cohen kappa 0.792; 14 of the 16 disagreements are definitions the reviewer marked as fitting no topic, and agreement is 0.98 without them.

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Majority class | 0.393 | 0.081 |
| Keyword baseline (zero-shot) | 0.536 | 0.648 |
| TF-IDF logistic regression | 0.807 | 0.825 |
| LoRA distilbert-base-uncased, mean of 3 seeds | 0.543 | 0.485 |

Fine-tuning did not beat the TF-IDF model, and on this split it did not beat the zero-shot keyword baseline on macro F1. LoRA (rank 8, 743,431 trainable of 67,702,286 parameters, 8 epochs, about 7.5 minutes per seed on CPU) is a result for a small CPU budget and 441 training clauses, not a verdict on the method. Over repeated group splits TF-IDF averaged accuracy 0.746 and macro F1 0.769, the keyword baseline 0.551 and 0.623. The smallest classes have 2 to 6 test clauses, so per-class results are noisy; see `examples/output/classifier_eval.md` for confusion matrices.

## Validation report

`docs/validation_report.md` applies the SR 26-2 headings (conceptual soundness, outcomes analysis, ongoing monitoring, governance and so on) to the assistant and ends with a findings table. Main findings: the validation is not independent; a plausible false passage in the corpus is not detected; answer selection was tuned after reading evaluation failures, so answer metrics are optimistic; abstention separates answerable from unanswerable questions imperfectly.

## Limitations

- Small evaluation set (66 questions, one author), wide intervals.
- The extractive answerer picks sentences, cannot combine passages and often cites the right paragraph with the wrong sentence (9 of the 12 answered misses).
- The sentence ranking and list handling were changed after looking at evaluation failures, so answer metrics are in-sample for that design.
- The corpus is a snapshot with partial extracts of several rules; the FR 2052a instructions are the 2022 version. Tables from eCFR are flattened and some cell boundaries merge words.
- Classifier labels are structural proxies, documentation and ongoing monitoring are folded into governance and validation, and the label review was done by one AI reviewer.
- LoRA was trained at laptop scale only.
- No generative model was run.

## Layout

```
src/reg_rag/        corpus handling, chunking, BM25, dense, retriever, answerer, injection, grounding, metrics, evaluation, classification/
data/corpus/        regulatory texts and manifest.json
data/eval/          questions.jsonl
data/classification clause dataset and the 100-clause hand review
examples/           evaluation scripts and their outputs
scripts/            corpus fetch script
docs/               validation report
tests/              unit tests (synthetic corpus, no network)
```
