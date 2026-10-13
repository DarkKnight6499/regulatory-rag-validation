# Clause topic classification results

727 clauses, labels: {'liquidity_buffer': 278, 'capital': 221, 'reporting': 141, 'liquidity_risk_management': 28, 'model_governance': 27, 'stress_testing': 22, 'model_validation': 10}.

## Label quality (hand review of 100 clauses)

Agreement 0.84 (Cohen kappa 0.792). The reviewer used a `none` label for 14 clauses that fit no topic (definitions); agreement excluding those is 0.98. Disagreements: 16, by heuristic label {'liquidity_buffer': 12, 'model_governance': 4}.

## Split

Sizes {'train': 441, 'val': 146, 'test': 140}. Label counts per split: {'train': {'liquidity_buffer': 167, 'capital': 135, 'reporting': 85, 'liquidity_risk_management': 19, 'model_governance': 16, 'stress_testing': 13, 'model_validation': 6}, 'val': {'liquidity_buffer': 56, 'capital': 45, 'reporting': 28, 'liquidity_risk_management': 5, 'stress_testing': 5, 'model_governance': 5, 'model_validation': 2}, 'test': {'liquidity_buffer': 55, 'capital': 41, 'reporting': 28, 'model_governance': 6, 'liquidity_risk_management': 4, 'stress_testing': 4, 'model_validation': 2}}.

## Test results (primary split, test set only)

| Model | Accuracy (95% CI) | Macro F1 (95% CI) |
|---|---|---|
| Majority class (liquidity_buffer) | 0.393 [0.314, 0.479] | 0.081 [0.068, 0.103] |
| Keyword baseline (zero-shot) | 0.536 [0.457, 0.614] | 0.648 [0.493, 0.729] |
| TF-IDF logistic regression (C=0.3) | 0.807 [0.743, 0.864] | 0.825 [0.690, 0.920] |
| LoRA distilbert-base-uncased, seed 0 | 0.586 [0.500, 0.664] | 0.513 [0.416, 0.591] |
| LoRA mean over 3 seeds | 0.543 | 0.485 |

Repeated group splits (mean and sd over split seeds; test sets are small and differ by seed):

| Model | Accuracy | Macro F1 |
|---|---|---|
| keyword_baseline | 0.551 (sd 0.136) | 0.623 (sd 0.117) |
| tfidf_lr | 0.746 (sd 0.085) | 0.769 (sd 0.109) |

LoRA per-seed test accuracy ['0.586', '0.486', '0.557'], macro F1 ['0.513', '0.440', '0.502'], best epochs [8, 5, 7], trainable parameters 743,431 of 67,702,286, training seconds [452.7, 435.2, 460.1]. Config: {'r': 8, 'alpha': 16, 'dropout': 0.1, 'targets': ['q_lin', 'v_lin']}, learning rate 0.0005, 8 epochs.

### Confusion matrix: Keyword baseline (rows true, columns predicted)

| true \ pred | model_governance | model_validation | liquidity_buffer | liquidity_risk_management | stress_testing | capital | reporting |
|---|---|---|---|---|---|---|---|
| model_governance | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| model_validation | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| liquidity_buffer | 35 | 0 | 19 | 0 | 0 | 1 | 0 |
| liquidity_risk_management | 0 | 0 | 0 | 3 | 1 | 0 | 0 |
| stress_testing | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| capital | 11 | 0 | 0 | 0 | 0 | 30 | 0 |
| reporting | 7 | 0 | 8 | 0 | 0 | 0 | 13 |

### Confusion matrix: TF-IDF logistic regression (rows true, columns predicted)

| true \ pred | model_governance | model_validation | liquidity_buffer | liquidity_risk_management | stress_testing | capital | reporting |
|---|---|---|---|---|---|---|---|
| model_governance | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| model_validation | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| liquidity_buffer | 0 | 0 | 42 | 0 | 0 | 12 | 1 |
| liquidity_risk_management | 0 | 0 | 0 | 3 | 1 | 0 | 0 |
| stress_testing | 0 | 0 | 1 | 1 | 2 | 0 | 0 |
| capital | 0 | 0 | 6 | 0 | 0 | 35 | 0 |
| reporting | 0 | 0 | 5 | 0 | 0 | 0 | 23 |

### Confusion matrix: LoRA seed 0 (rows true, columns predicted)

| true \ pred | model_governance | model_validation | liquidity_buffer | liquidity_risk_management | stress_testing | capital | reporting |
|---|---|---|---|---|---|---|---|
| model_governance | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| model_validation | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| liquidity_buffer | 0 | 0 | 20 | 4 | 0 | 31 | 0 |
| liquidity_risk_management | 0 | 0 | 3 | 0 | 0 | 1 | 0 |
| stress_testing | 0 | 0 | 0 | 1 | 0 | 2 | 1 |
| capital | 0 | 0 | 4 | 0 | 1 | 35 | 1 |
| reporting | 0 | 0 | 7 | 0 | 0 | 1 | 20 |
