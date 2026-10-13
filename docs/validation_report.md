# Validation report: regulatory question-answering assistant (reg_rag)

Structured after the headings of SR 26-2 (interagency Supervisory Guidance on Model Risk Management, April 17, 2026, which replaced SR 11-7 and SR 21-8), whose text is in this repository's corpus (`data/corpus/sr26-2.md`). The validator and the developer are the same party, so this is a self-assessment and does not meet the guidance's expectation of effective challenge by objective, independent experts (finding F1). All numbers come from `examples/output/rag_eval.md`, `examples/output/classifier_eval.md` and the JSON files beside them, produced by the scripts in `examples/`.

## 1. Executive summary

The assistant retrieves and quotes passages from eight US bank regulatory documents (764 passages) and abstains when evidence is weak. On 52 answerable questions, hybrid retrieval puts a gold passage first for 0.731 (BM25 alone 0.712) and in the top 5 for 0.962 (BM25 0.962). The extractive answerer contains the gold answer span in 0.635 of answerable questions overall, and cites a correct passage first for 0.822 of the questions it answers. Every returned sentence was found verbatim in the retrieved passages (groundedness 1.000), and a deliberate change to one figure or modal verb was caught in 30 of 30 applicable answers. Prompt-injection text in a question or in the corpus is largely neutralised by a pattern defence (instruction-style poisoning changes behaviour for 1 of 52 questions with the defence versus 52 of 52 without), but a poisoned passage that makes a false claim in plain regulatory language is not detected (8 of 52 answers contain the false claim, with or without the defence).

Overall rating: usable as a citation finder for a human reader, not usable as an unattended answering system. Key reasons: small evaluation set (66 questions), answer selection tuned after seeing evaluation failures, an abstention rule that trades 0.500 precision against 0.889 recall, and no defence against plausible false text in the corpus.

## 2. Model description

| Component | Design |
|---|---|
| Corpus | 8 documents: 12 CFR 249 (LCR), 12 CFR 252 subparts D and E, 12 CFR 217 subparts B, C and section 217.30, SR 26-2 (attachment), FR 2052a instructions (version dated 2022-04-29). eCFR text pinned to the 2026-10-01 version; retrieval date 2026-10-13; SHA-256 per file in `data/corpus/manifest.json`. |
| Chunking | By legal structure: CFR section, then paragraph tree (marker levels a, 1, i, A, I); guidance by section, subheading and paragraph. Chunks of 220 to 1,500 characters. Each chunk carries a citation string and the list of paragraph keys it covers. |
| Retrieval | Own BM25 (k1 1.5, b 0.75); dense retrieval with bge-small-en-v1.5 (query prefix) or all-MiniLM-L6-v2; hybrid by reciprocal rank fusion (k 60). |
| Answering | Extractive. Confidence is 0.6 x idf-weighted query-term coverage plus 0.4 x dense cosine of the best of the top 3 passages. Sentences are ranked by 0.5 x coverage plus 0.5 x min-max scaled embedding similarity, decayed 0.92 per passage rank; the top 2 are returned with citations. A sentence ending in a colon is extended with the following list items. Below the threshold the assistant abstains. |
| Generative hook | `GenerativeAnswerer` accepts any text generator; a generation that is not grounded in the retrieved passages or looks like an instruction is replaced by the extractive answer. Tested only with a stub; no API model was run. |
| Injection defence | Regex patterns for instruction-like text; matching sentences are removed from the question, matching passages and sentences are dropped. |
| Clause classifier | Topic labels by document and section (7 classes). Compared: keyword baseline, TF-IDF logistic regression, LoRA on distilbert-base-uncased. |

Materiality (SR 26-2, section III): exposure is low because no business decision or reported figure depends on the output, and purpose is informational, so inherent model risk is low as long as a human reads the cited paragraph. It rises sharply if answers are used unattended or in a regulatory filing.

Scope note: the assistant is extractive, so no generative component is in use. Generative and agentic AI models are outside SR 26-2 (footnote 3), which matters if the `GenerativeAnswerer` hook were connected to an LLM; the guidance then points to the bank's general risk management and governance practices instead.

Intended use: help a reader locate and quote the controlling text. Not intended for legal or compliance determinations.

## 3. Conceptual soundness

- Chunking on legal structure keeps each answer attributable to a paragraph, which is what a regulatory reader needs. Weakness: chunks that merge several short paragraphs give coarser citations, and 12 CFR 249 definitions are chunked by term, so cross-references between paragraphs are not followed.
- BM25 is a sound first stage for text full of defined terms and numbers; dense retrieval adds recall on paraphrased questions. The evaluation shows the two are close (section 5).
- Extractive answering cannot invent text, which bounds the hallucination risk by construction. The cost is that it cannot combine passages, compare or compute; questions needing synthesis score as misses.
- The confidence score mixes lexical and embedding evidence with hand-set weights (0.6 and 0.4). The weights were chosen by inspection and not tuned on held-out data.
- The groundedness check is a verbatim containment test. It proves the text exists in a retrieved passage, not that it answers the question.

## 4. Developmental evidence

- Data: corpus manifest with hashes (`verify_manifest` is tested). 66 evaluation questions written by reading the passages: 52 answerable (each with gold paragraph keys and an answer span, validated against the corpus by `validate_questions`), 8 unanswerable, 5 adversarial with a clean twin, 1 pure-injection question that should be refused.
- Development process: the answerer's sentence ranking and list extension were changed after reading evaluation failures. The evaluation questions therefore are not a clean test of the final answerer; the numbers in section 5 are optimistic by an unknown amount (finding F3). The abstention threshold was tuned on the even-indexed half of the questions and reported on the odd-indexed half.
- Classifier labels come from transparent structural rules (document and section, with SR 26-2 section V as validation), not from the words of each clause. A hand review of 100 clauses by a single AI reviewer gave agreement 0.84 (Cohen kappa 0.792). 14 of the 16 disagreements are definitions that the reviewer marked as fitting no topic; excluding them agreement is 0.98. The model_validation class has only 10 clauses (2 in the test set). Human review has not been done (finding F6).
- Tests: unit tests on a synthetic corpus run with no network and no model downloads. Model-dependent steps run through `examples/` scripts.

## 5. Outcomes analysis

### Retrieval (52 answerable questions, bootstrap 95% intervals)

| Retriever | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.712 | 0.962 | 0.810 |
| Dense (bge-small) | 0.769 | 0.923 | 0.817 |
| Hybrid (BM25 + bge-small) | 0.731 | 0.962 | 0.828 |
| Dense (MiniLM-L6) | 0.673 | 0.885 | 0.775 |
| Hybrid (BM25 + MiniLM-L6) | 0.769 | 0.981 | 0.854 |

Paired MRR differences: hybrid minus BM25 0.018 (CI -0.057 to 0.092), hybrid minus dense 0.010 (CI -0.038 to 0.057), BM25 minus dense -0.008 (CI -0.103 to 0.092); none is distinguishable from zero. The hybrid with MiniLM has the best point estimate but was one of five configurations compared, and its interval overlaps the others.

### Answers and abstention

Threshold 0.64 tuned on 33 dev questions (balanced abstention accuracy 0.931) and scored on the 33 held-out questions (0.829).

| Metric | Held-out half | All questions (threshold tuned on half) | No abstention |
|---|---|---|---|
| Answer span contained, all answerable | 0.577 | 0.635 | 0.692 |
| Answer span contained, when answered | 0.682 | 0.733 | 0.692 |
| Top citation correct, when answered | 0.773 | 0.822 | 0.750 |
| Groundedness (verbatim) | 1.000 | 1.000 | 1.000 |
| Abstention precision | 0.500 | 0.500 | 1.000 |
| Abstention recall | 0.800 | 0.889 | 0.111 |

Abstention is the weakest control. Confidence for unanswerable questions ranges from 0.356 to 0.653 and for answerable ones from 0.407 to 0.959, so the ranges overlap. At threshold 0.64, 4 of the 26 answerable held-out questions are wrongly refused (Q06, Q22, Q28, Q32) and one unanswerable question (U08, confidence 0.653) is answered. Raising the threshold to 0.8 cuts answer-span containment to 0.404.

Of the 45 answerable questions answered at threshold 0.64, 33 contain the gold span; 9 of the 12 misses cite a correct passage but select the wrong sentence, so the retrieval is better than the answer selection.

### Prompt injection

| Test | Defence off | Defence on |
|---|---|---|
| Question carries an instruction, payload appears in answer | 0 of 5 | 0 of 5 |
| Same answer as the question's clean twin | 0 of 5 | 4 of 5 |
| Pure-injection question refused | 1 of 1 | 1 of 1 |
| Corpus poisoning, instruction style: behaviour changed | 52 of 52 | 1 of 52 |
| Corpus poisoning, embedded instruction: behaviour changed / payload in answer | 29 of 52 / 0 | 2 of 52 / 0 |
| Corpus poisoning, fake claim without instruction language: behaviour changed / payload in answer | 19 of 52 / 8 | 19 of 52 / 8 |

The extractive answerer never executes an instruction, so the payload seldom reaches an answer, but without the defence a poisoned passage displaces the right evidence (the poison passage is cited in 52 of 52 instruction-style cases). The defence works against instruction wording and cannot work against a false statement written as ordinary regulatory text.

### Clause classifier (test set of 140 clauses, split by document section)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.393 | 0.081 |
| Keyword baseline (zero-shot) | 0.536 | 0.648 |
| TF-IDF logistic regression | 0.807 | 0.825 |
| LoRA distilbert-base-uncased (3 seeds, mean) | 0.543 | 0.485 |

LoRA did not beat TF-IDF, and its macro F1 is below the zero-shot keyword baseline. Best validation epochs were 8, 5 and 7 of 8, so the comparison is for a short CPU budget (about 7.5 minutes per seed, 743,431 trainable parameters of 67,702,286), not a ceiling for the method. The keyword baseline has a macro F1 0.163 higher than LoRA at a similar accuracy, because it sends many clauses (55 of 140) to model_governance. With 2 test clauses in model_validation and 4 to 6 in other small classes, per-class scores are noisy.

## 6. Ongoing monitoring

Not yet operating. What exists and what a deployment would need:

- Exists: corpus hashes detect silent file changes; `examples/run_rag_eval.py` reruns all metrics in about 20 seconds once embeddings are cached; the eCFR URL carries a date, so a refetch with a newer date gives a diff of the regulation.
- Needed: a scheduled refetch and evaluation rerun, alerting on a drop in recall@5 or a rise in ungrounded answers, review of abstention rate on live questions, and a procedure to add questions when a rule changes. The evaluation questions pin gold citations to paragraph keys, so a renumbered rule breaks them visibly (the validator reports unknown keys).

## 7. Limitations

- 66 questions and one author; intervals are wide (for example held-out abstention precision 0.571 with interval 0.200 to 1.000).
- Answer selection was tuned after seeing failures; the reported answer metrics are in-sample for that design choice.
- Extractive answers cannot synthesise across passages and often return the right paragraph with the wrong sentence.
- Regulations change. The corpus is a snapshot (eCFR 2026-10-01); the FR 2052a instructions are the 2022-04-29 form version and may not be current. SR 26-2 is the April 2026 guidance and is non-enforceable supervisory guidance.
- Several sections are partial extracts (12 CFR 252 subparts D and E, 217 subparts B and C, one section of 217 subpart D); answers about other parts of these rules will abstain or mislead.
- Table content from eCFR is flattened to text and cell boundaries can merge words (for example "Maximumpayout ratio"); tables are poorly answerable.
- Classifier labels are structural proxies for topic, `documentation` and `ongoing monitoring` are folded into model governance and model validation, and the smallest classes have 2 to 6 test clauses (model_validation has 10 clauses in total).
- Not legal advice.

## 8. Findings and recommendations

| ID | Finding | Severity | Recommendation |
|---|---|---|---|
| F1 | Validation is performed by the developer, not independently (SR 26-2 expects effective challenge by objective experts). | High | Have an independent reviewer rerun the scripts and write their own questions before any real use. |
| F2 | Plausible false text in the corpus is neither detected nor flagged (8 of 52 answers repeat the fake claim). | High | Restrict the index to hash-verified primary sources and compare chunks against the manifest on load; add a cross-source consistency check for numeric claims. |
| F3 | Answer selection was tuned on the evaluation set, so answer metrics are optimistic. | Medium | Write a fresh question set of at least 100 items, freeze the answerer, and report once. |
| F4 | Abstention separates answerable from unanswerable questions imperfectly (precision 0.500, recall 0.889 at threshold 0.64). | Medium | Show the confidence to the reader and return the best passages as "possibly relevant" instead of a hard refusal; collect more unanswerable questions to tune the threshold. |
| F5 | Retrieval is good (recall@5 0.885 to 0.981) but sentence selection misses the gold span for 12 of 45 answered questions. | Medium | Return the whole cited paragraph alongside the selected sentences; evaluate a trained sentence ranker or an API generator behind the existing grounded hook. |
| F6 | Classifier label quality was reviewed by one AI reviewer only. | Low | Human review of the 100-clause sample and of the 14 disagreeing definitions. |
| F7 | Corpus is a dated snapshot with partial extracts and possibly outdated FR 2052a instructions. | Medium | Schedule refetch with hash diff; state the snapshot date in every answer. |
| F8 | LoRA fine-tuning underperformed TF-IDF under the CPU budget used. | Info | Prefer TF-IDF for this dataset size; revisit LoRA only with more labelled data and longer training. |

## 9. Governance and use conditions

- Roles (SR 26-2, section VI): developer and validator are one person. Before any use beyond demonstration, an independent reviewer should own the validation and an owner should be named for the model inventory entry.
- Inventory and documentation: record the assistant as a low-materiality model, with its corpus snapshot date, thresholds and evaluation set version; this report and the README are the documentation.
- Third-party components (section VII): bge-small-en-v1.5, all-MiniLM-L6-v2 and distilbert-base-uncased are vendor or open-source models used as supplied. Their training data and internal behaviour were not validated; only their outputs on this corpus were tested.
- Use conditions: a human reads the cited paragraph; low-confidence answers are shown as possibly relevant, not final; no use in filings or compliance determinations.
- Change triggers: a new corpus snapshot, a changed threshold, a new embedding model or a connected generator each require rerunning the evaluation and updating this report.

## 10. Conclusion

The retrieval and citation layer is adequate for helping a reader find the controlling paragraph, and the extractive design removes fabricated content. The assistant should not be used unattended: abstention is unreliable, answer selection is unvalidated on fresh questions, and corpus poisoning with plausible text is not defended. Findings F1 to F3 should be closed before any use beyond demonstration.
