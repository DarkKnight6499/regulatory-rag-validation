# Clause topic classification results

773 clauses, labels: {'liquidity_buffer': 278, 'capital': 221, 'reporting': 141, 'model_governance': 49, 'model_validation': 34, 'liquidity_risk_management': 28, 'stress_testing': 22}.

## Label quality (hand review of 100 clauses)

Agreement 0.88 (Cohen kappa 0.844). The reviewer used a `none` label for 12 clauses that fit no topic (definitions); agreement excluding those is 1.00. Disagreements: 12, by heuristic label {'liquidity_buffer': 12}.

## Split

Sizes {'train': 469, 'val': 150, 'test': 154}. Label counts per split: {'train': {'liquidity_buffer': 167, 'capital': 135, 'reporting': 85, 'model_governance': 29, 'model_validation': 21, 'liquidity_risk_management': 19, 'stress_testing': 13}, 'val': {'liquidity_buffer': 55, 'capital': 41, 'reporting': 28, 'model_governance': 10, 'model_validation': 6, 'liquidity_risk_management': 5, 'stress_testing': 5}, 'test': {'liquidity_buffer': 56, 'capital': 45, 'reporting': 28, 'model_governance': 10, 'model_validation': 7, 'liquidity_risk_management': 4, 'stress_testing': 4}}.

## Test results (primary split, test set only)

| Model | Accuracy (95% CI) | Macro F1 (95% CI) |
|---|---|---|
| Majority class (liquidity_buffer) | 0.364 [0.286, 0.435] | 0.076 [0.063, 0.089] |
| Keyword baseline (zero-shot) | 0.539 [0.461, 0.617] | 0.604 [0.466, 0.701] |
| TF-IDF logistic regression (C=0.3) | 0.792 [0.727, 0.857] | 0.684 [0.563, 0.797] |
| LoRA distilbert-base-uncased, seed 0 | 0.649 [0.571, 0.721] | 0.480 [0.395, 0.564] |
| LoRA mean over 3 seeds | 0.667 | 0.512 |

Repeated group splits (mean and sd over split seeds; test sets are small and differ by seed):

| Model | Accuracy | Macro F1 |
|---|---|---|
| keyword_baseline | 0.559 (sd 0.079) | 0.613 (sd 0.061) |
| tfidf_lr | 0.726 (sd 0.054) | 0.692 (sd 0.034) |

LoRA per-seed test accuracy ['0.649', '0.708', '0.643'], macro F1 ['0.480', '0.554', '0.500'], best epochs [8, 8, 6], trainable parameters 743,431 of 67,702,286, training seconds [455.9, 406.4, 400.7]. Config: {'r': 8, 'alpha': 16, 'dropout': 0.1, 'targets': ['q_lin', 'v_lin']}, learning rate 0.0005, 8 epochs.

### Confusion matrix: Keyword baseline (rows true, columns predicted)

| true \ pred | model_governance | model_validation | liquidity_buffer | liquidity_risk_management | stress_testing | capital | reporting |
|---|---|---|---|---|---|---|---|
| model_governance | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| model_validation | 4 | 3 | 0 | 0 | 0 | 0 | 0 |
| liquidity_buffer | 32 | 0 | 22 | 0 | 0 | 1 | 1 |
| liquidity_risk_management | 0 | 0 | 0 | 3 | 1 | 0 | 0 |
| stress_testing | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| capital | 23 | 0 | 0 | 0 | 0 | 22 | 0 |
| reporting | 3 | 0 | 3 | 0 | 1 | 0 | 21 |

### Confusion matrix: TF-IDF logistic regression (rows true, columns predicted)

| true \ pred | model_governance | model_validation | liquidity_buffer | liquidity_risk_management | stress_testing | capital | reporting |
|---|---|---|---|---|---|---|---|
| model_governance | 9 | 1 | 0 | 0 | 0 | 0 | 0 |
| model_validation | 2 | 5 | 0 | 0 | 0 | 0 | 0 |
| liquidity_buffer | 0 | 0 | 42 | 0 | 0 | 10 | 4 |
| liquidity_risk_management | 0 | 0 | 0 | 1 | 3 | 0 | 0 |
| stress_testing | 0 | 0 | 1 | 1 | 2 | 0 | 0 |
| capital | 0 | 0 | 5 | 0 | 0 | 40 | 0 |
| reporting | 0 | 0 | 2 | 0 | 2 | 1 | 23 |

### Confusion matrix: LoRA seed 0 (rows true, columns predicted)

| true \ pred | model_governance | model_validation | liquidity_buffer | liquidity_risk_management | stress_testing | capital | reporting |
|---|---|---|---|---|---|---|---|
| model_governance | 8 | 2 | 0 | 0 | 0 | 0 | 0 |
| model_validation | 4 | 3 | 0 | 0 | 0 | 0 | 0 |
| liquidity_buffer | 0 | 0 | 36 | 0 | 0 | 19 | 1 |
| liquidity_risk_management | 0 | 0 | 3 | 0 | 0 | 1 | 0 |
| stress_testing | 0 | 0 | 0 | 1 | 0 | 2 | 1 |
| capital | 0 | 0 | 20 | 0 | 0 | 25 | 0 |
| reporting | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
