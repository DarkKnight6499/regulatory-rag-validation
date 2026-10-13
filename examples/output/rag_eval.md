# RAG evaluation results

Passages 764, questions 66 ({'answerable': 52, 'unanswerable': 8, 'adversarial': 5, 'adversarial_abstain': 1}), bootstrap 2000 resamples, seed 0.

## Retrieval (answerable questions, n=52)

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.712 [0.577, 0.827] | 0.885 [0.788, 0.962] | 0.962 [0.904, 1.000] | 0.981 [0.942, 1.000] | 0.810 [0.722, 0.892] |
| Dense (bge-small) | 0.769 [0.654, 0.885] | 0.827 [0.712, 0.923] | 0.923 [0.846, 0.981] | 0.942 [0.885, 1.000] | 0.817 [0.722, 0.906] |
| Hybrid RRF (BM25 + bge-small) | 0.731 [0.615, 0.846] | 0.904 [0.827, 0.981] | 0.962 [0.904, 1.000] | 0.981 [0.942, 1.000] | 0.828 [0.750, 0.909] |
| Dense (MiniLM-L6) | 0.673 [0.538, 0.808] | 0.846 [0.750, 0.942] | 0.885 [0.788, 0.962] | 0.981 [0.942, 1.000] | 0.775 [0.685, 0.862] |
| Hybrid RRF (BM25 + MiniLM-L6) | 0.769 [0.654, 0.885] | 0.923 [0.846, 0.981] | 0.981 [0.942, 1.000] | 0.981 [0.942, 1.000] | 0.854 [0.774, 0.926] |

Paired bootstrap difference in reciprocal rank (mean, 95% CI):

- hybrid_vs_bm25: 0.018 [-0.057, 0.092]
- hybrid_vs_dense: 0.010 [-0.038, 0.057]
- bm25_vs_dense: -0.008 [-0.103, 0.092]

## Extractive answerer

Abstention threshold 0.64 tuned on the 33 even-indexed questions; reported on the 33 odd-indexed (held-out) questions. Balanced abstention accuracy: dev 0.931, test 0.829.

| Metric | Held-out half | All questions (threshold tuned on half of them) | No abstention |
|---|---|---|---|
| Answer span contained (all answerable) | 0.577 [0.385, 0.769] | 0.635 [0.500, 0.769] | 0.692 [0.558, 0.808] |
| Answer span contained (when answered) | 0.682 [0.500, 0.864] | 0.733 [0.600, 0.844] | 0.692 [0.558, 0.808] |
| Token F1 (best sentence, all answerable) | 0.361 [0.266, 0.461] | 0.407 [0.325, 0.484] | 0.469 [0.393, 0.546] |
| Top citation correct (when answered) | 0.773 [0.591, 0.955] | 0.822 [0.711, 0.911] | 0.750 [0.635, 0.865] |
| Any citation correct (when answered) | 0.909 [0.773, 1.000] | 0.911 [0.822, 0.978] | 0.846 [0.750, 0.942] |
| Groundedness (sentences found verbatim in retrieved passages) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Abstention precision | 0.500 [0.143, 0.875] | 0.500 [0.250, 0.750] | 1.000 [1.000, 1.000] |
| Abstention recall | 0.800 [0.333, 1.000] | 0.889 [0.636, 1.000] | 0.111 [0.000, 0.375] |

Counts on held-out half: 33 questions, 26 answerable, 22 answered; abstention tp/fp/fn = 4/4/1.

### Threshold sweep (all questions)

| Threshold | Abstain precision | Abstain recall | Span contained, all answerable |
|---|---|---|---|
| 0.0 | 1.000 | 0.111 | 0.692 |
| 0.3 | 1.000 | 0.111 | 0.692 |
| 0.4 | 1.000 | 0.333 | 0.692 |
| 0.5 | 0.750 | 0.667 | 0.673 |
| 0.6 | 0.583 | 0.778 | 0.654 |
| 0.7 | 0.391 | 1.000 | 0.596 |
| 0.8 | 0.231 | 1.000 | 0.404 |

Groundedness negative control: one figure or modal verb altered in the first returned sentence of 30 answers (of 50 answered) was detected in 30.

## Prompt injection

User-side: adversarial questions (instruction plus a real question) against their clean twins.

| Defence | Same answer as clean twin | Payload in answer | Pure-injection question abstained |
|---|---|---|---|
| False | 0/5 | 0/5 | 1/1 |
| True | 4/5 | 0/5 | 1/1 |

Corpus-side: one poisoned passage per answerable question added to the index (behaviour compared with the same defence setting on the clean index).

| Poison style | Defence | Behaviour changed | Payload in answer | Poison passage cited | Span contained under attack |
|---|---|---|---|---|---|
| instruction | False | 52/52 | 0 | 52 | 0.558 |
| instruction | True | 1/52 | 0 | 0 | 0.635 |
| embedded | False | 29/52 | 0 | 27 | 0.577 |
| embedded | True | 2/52 | 0 | 0 | 0.615 |
| fakeclaim | False | 19/52 | 8 | 18 | 0.577 |
| fakeclaim | True | 19/52 | 8 | 18 | 0.577 |
