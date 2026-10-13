# regulatory-rag-validation

A retrieval-augmented question-answering assistant over primary US bank regulatory texts (liquidity, capital and model risk), with an evaluation harness, a clause-classifier comparison (keyword baseline, TF-IDF logistic regression, LoRA fine-tuning) and an SR 11-7 style validation report of the assistant itself.

The assistant is extractive: it returns cited sentences from the corpus and abstains when evidence is weak. A generative step can be plugged in behind an interface, but none is needed and no API key is used anywhere.

Not legal advice. The corpus is a dated snapshot of rules that change.

## Corpus

810 passages from 8 documents, about 520 KB, each file hashed in `data/corpus/manifest.json` (source URL, retrieval date 2026-10-12, SHA-256).

| Document | Source |
|---|---|
| 12 CFR 249 (Liquidity Coverage Ratio) | eCFR, version 2026-10-01 |
| 12 CFR 252 subparts D and E (liquidity risk management, stress testing) | eCFR, version 2026-10-01 |
| 12 CFR 217 subparts B and C, section 217.30 (capital) | eCFR, version 2026-10-01 |
| SR 11-7 attachment (model risk management guidance) | federalreserve.gov PDF |
| FR 2052a instructions, general instructions and field definitions (form version dated 2022-04-29, possibly not current) | federalreserve.gov PDF |

Chunking follows legal structure (section, then paragraph tree), and every passage carries a citation string and the paragraph keys it covers, for example `12 CFR 249.10(a)` or `SR 11-7, V (3. Outcomes Analysis), para 4`.

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
answer = answerer.answer("What is benchmarking of a model?")
print(answer.text, answer.citations)
```

To use a generative model, wrap any `prompt -> text` function in `CallableGenerator` and pass it to `GenerativeAnswerer`. A generation that is not grounded in the retrieved passages, or that looks like an instruction, is replaced by the extractive answer. This hook is tested only with a stub function.

## Evaluation set

66 questions written from the corpus text, each answerable one with gold paragraph keys and an answer span that is checked against the corpus by `validate_questions`: 52 answerable, 8 unanswerable (should abstain), 5 adversarial (a real question plus an injected instruction, each with a clean twin), 1 pure-injection question (should be refused). Intervals are 95% percentile bootstrap, 2,000 resamples, seed 0.

## Results

### Retrieval, 52 answerable questions

| Retriever | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.731 | 0.981 | 0.823 |
| Dense, bge-small | 0.692 | 0.923 | 0.776 |
| Hybrid (RRF), BM25 + bge-small | 0.750 | 0.962 | 0.840 |
| Dense, MiniLM-L6 | 0.654 | 0.904 | 0.760 |
| Hybrid (RRF), BM25 + MiniLM-L6 | 0.846 | 0.981 | 0.893 |

Intervals are wide (for example BM25 MRR 0.823, CI 0.734 to 0.902). Paired MRR difference, hybrid minus BM25: 0.018 (CI -0.064 to 0.098), so on this corpus BM25 is not distinguishable from the hybrid. Hybrid minus dense: 0.065 (CI 0.032 to 0.103).

### Answers and abstention

The abstention threshold (0.64) was tuned on the 33 even-indexed questions and reported on the 33 odd-indexed ones.

| Metric | Held-out half | All questions | No abstention |
|---|---|---|---|
| Answer span contained, all answerable | 0.654 | 0.692 | 0.692 |
| Answer span contained, when answered | 0.739 | 0.766 | 0.692 |
| Top citation correct, when answered | 0.826 | 0.830 | 0.750 |
| Groundedness (every sentence verbatim in retrieved passages) | 1.000 | 1.000 | 1.000 |
| Abstention precision | 0.571 | 0.571 | 1.000 |
| Abstention recall | 0.800 | 0.889 | 0.111 |

Balanced abstention accuracy was 0.948 on the dev half and 0.846 on the held-out half. Abstention is the weakest control: confidence for unanswerable questions (0.349 to 0.649) overlaps that of answerable ones (0.387 to 0.973). Changing one figure or modal verb in a returned sentence was caught by the groundedness check in 28 of 28 applicable answers.

### Prompt injection

| Test | Defence off | Defence on |
|---|---|---|
| Instruction in the question: payload appears in answer | 0 of 5 | 0 of 5 |
| Same answer as the question's clean twin | 0 of 5 | 4 of 5 |
| Corpus poisoned, instruction style: behaviour changed | 52 of 52 | 0 of 52 |
| Corpus poisoned, instruction embedded in a copy of the gold passage: behaviour changed (payload in answer) | 31 of 52 (2) | 1 of 52 (0) |
| Corpus poisoned with a false claim and no instruction wording: behaviour changed (payload in answer) | 20 of 52 (10) | 20 of 52 (10) |

The pattern defence stops instruction-style attacks and cannot stop a plausible false statement, which is the main open security finding.

### Clause topic classifier

773 clauses in 7 topics, labelled by document and section (structural rules), split by section so no section spans two splits (train 469, validation 150, test 154). Hand review of 100 clauses (a single AI reviewer): agreement 0.88, Cohen kappa 0.844; all 12 disagreements are definitions the reviewer marked as fitting no topic, and agreement is 1.00 without them.

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Majority class | 0.364 | 0.076 |
| Keyword baseline (zero-shot) | 0.539 | 0.604 |
| TF-IDF logistic regression | 0.792 | 0.684 |
| LoRA distilbert-base-uncased, mean of 3 seeds | 0.667 | 0.512 |

Fine-tuning did not beat the TF-IDF model. LoRA (rank 8, 743,431 trainable of 67,702,286 parameters, 8 epochs, about 7 minutes per seed on CPU) was still improving when it stopped, so this is a result for a small CPU budget and 469 training clauses, not a verdict on the method. Over 5 repeated splits TF-IDF averaged accuracy 0.726 and macro F1 0.692, the keyword baseline 0.559 and 0.613. The smallest classes have 4 to 7 test clauses, so per-class results are noisy; see `examples/output/classifier_eval.md` for confusion matrices.

## Validation report

`docs/validation_report.md` applies the SR 11-7 headings (conceptual soundness, ongoing monitoring, outcomes analysis and so on) to the assistant and ends with a findings table. Main findings: the validation is not independent; a plausible false passage in the corpus is not detected; answer selection was tuned after reading evaluation failures, so answer metrics are optimistic; abstention separates answerable from unanswerable questions imperfectly.

## Limitations

- Small evaluation set (66 questions, one author), wide intervals.
- The extractive answerer picks sentences, cannot combine passages and often cites the right paragraph with the wrong sentence (8 of the 11 answered misses).
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
