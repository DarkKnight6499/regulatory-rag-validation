# RAG evaluation results

Passages 810, questions 66 ({'answerable': 52, 'unanswerable': 8, 'adversarial': 5, 'adversarial_abstain': 1}), bootstrap 2000 resamples, seed 0.

## Retrieval (answerable questions, n=52)

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.731 [0.596, 0.846] | 0.904 [0.808, 0.981] | 0.981 [0.942, 1.000] | 0.981 [0.942, 1.000] | 0.823 [0.734, 0.902] |
| Dense (bge-small) | 0.692 [0.558, 0.808] | 0.827 [0.712, 0.923] | 0.923 [0.846, 0.981] | 0.942 [0.885, 1.000] | 0.776 [0.679, 0.864] |
| Hybrid RRF (BM25 + bge-small) | 0.750 [0.635, 0.865] | 0.904 [0.827, 0.981] | 0.962 [0.904, 1.000] | 0.981 [0.942, 1.000] | 0.840 [0.765, 0.913] |
| Dense (MiniLM-L6) | 0.654 [0.519, 0.769] | 0.846 [0.750, 0.942] | 0.904 [0.827, 0.981] | 0.981 [0.942, 1.000] | 0.760 [0.667, 0.846] |
| Hybrid RRF (BM25 + MiniLM-L6) | 0.846 [0.750, 0.942] | 0.923 [0.846, 0.981] | 0.981 [0.942, 1.000] | 0.981 [0.942, 1.000] | 0.893 [0.814, 0.958] |

Paired bootstrap difference in reciprocal rank (mean, 95% CI):

- hybrid_vs_bm25: 0.018 [-0.064, 0.098]
- hybrid_vs_dense: 0.065 [0.032, 0.103]
- bm25_vs_dense: 0.047 [-0.044, 0.140]

## Extractive answerer

Abstention threshold 0.64 tuned on the 33 even-indexed questions; reported on the 33 odd-indexed (held-out) questions. Balanced abstention accuracy: dev 0.948, test 0.846.

| Metric | Held-out half | All questions (threshold tuned on half of them) | No abstention |
|---|---|---|---|
| Answer span contained (all answerable) | 0.654 [0.462, 0.846] | 0.692 [0.558, 0.808] | 0.692 [0.558, 0.808] |
| Answer span contained (when answered) | 0.739 [0.565, 0.913] | 0.766 [0.638, 0.872] | 0.692 [0.558, 0.808] |
| Token F1 (best sentence, all answerable) | 0.434 [0.314, 0.556] | 0.460 [0.372, 0.546] | 0.477 [0.394, 0.558] |
| Top citation correct (when answered) | 0.826 [0.652, 0.957] | 0.830 [0.723, 0.936] | 0.750 [0.635, 0.865] |
| Any citation correct (when answered) | 0.913 [0.783, 1.000] | 0.915 [0.830, 0.979] | 0.846 [0.750, 0.942] |
| Groundedness (sentences found verbatim in retrieved passages) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Abstention precision | 0.571 [0.200, 1.000] | 0.571 [0.308, 0.833] | 1.000 [1.000, 1.000] |
| Abstention recall | 0.800 [0.333, 1.000] | 0.889 [0.636, 1.000] | 0.111 [0.000, 0.375] |

Counts on held-out half: 33 questions, 26 answerable, 23 answered; abstention tp/fp/fn = 4/3/1.

### Threshold sweep (all questions)

| Threshold | Abstain precision | Abstain recall | Span contained, all answerable |
|---|---|---|---|
| 0.0 | 1.000 | 0.111 | 0.692 |
| 0.3 | 1.000 | 0.111 | 0.692 |
| 0.4 | 0.800 | 0.444 | 0.692 |
| 0.5 | 0.750 | 0.667 | 0.692 |
| 0.6 | 0.636 | 0.778 | 0.692 |
| 0.7 | 0.474 | 1.000 | 0.654 |
| 0.8 | 0.231 | 1.000 | 0.423 |

Groundedness negative control: one figure or modal verb altered in the first returned sentence of 28 answers (of 52 answered) was detected in 28.

## Prompt injection

User-side: adversarial questions (instruction plus a real question) against their clean twins.

| Defence | Same answer as clean twin | Payload in answer | Pure-injection question abstained |
|---|---|---|---|
| False | 0/5 | 0/5 | 1/1 |
| True | 4/5 | 0/5 | 1/1 |

Corpus-side: one poisoned passage per answerable question added to the index (behaviour compared with the same defence setting on the clean index).

| Poison style | Defence | Behaviour changed | Payload in answer | Poison passage cited | Span contained under attack |
|---|---|---|---|---|---|
| instruction | False | 47/52 | 0 | 2 | 0.000 |
| instruction | True | 46/52 | 0 | 0 | 0.000 |
| embedded | False | 47/52 | 0 | 1 | 0.000 |
| embedded | True | 46/52 | 0 | 0 | 0.000 |
| fakeclaim | False | 47/52 | 0 | 1 | 0.000 |
| fakeclaim | True | 47/52 | 0 | 1 | 0.000 |
