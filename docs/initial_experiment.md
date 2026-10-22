# Initial Experiment: Model Reliability Analysis

**Project:** Student Social Media Addiction Score Prediction
**Contribution:** Himel (experimental evaluation and model reliability analysis)
**Notebook:** `notebooks/03_model_reliability_analysis.ipynb`
**Status:** The notebook was executed end to end (0 errors). Two runs gave identical numbers. Environment: Python 3.13.16, scikit-learn 1.9.1, XGBoost 3.4.1, pandas 3.0.5, numpy 2.5.3. Full list in `results/environment.json`.

Labels used in this document:

| Label | Meaning |
|---|---|
| **VERIFIED** | Produced by running the notebook |
| **RECORDED (unverified)** | Copied from an earlier notebook's saved output. Not re-run here. |

---

## 1. Purpose and distinct contribution

My teammates built the EDA and preprocessing notebook and the StablePFN model with its initial benchmark.
My contribution asks a different question: **are the reported regression results trustworthy, and how should models be compared fairly?**

| Already done by teammates | Added in this contribution |
|---|---|
| EDA plots, IQR outlier check | Data integrity audit (duplicates, ID leakage, conflicting labels) |
| One train/val/test split | Validation of that split, plus nested cross-validation on the development set |
| StablePFN training and one test score | Fair baselines on the **same** test rows, with bootstrap confidence intervals |
| Gate-score "feature importance" | Permutation importance (model reliance, explicitly non-causal) |
| Not done | Leakage tests: shuffled-target test, single-feature test, Country encoding comparison |
| Not done | Reliability tests: CV vs test, overfitting gap, learning curves, split sensitivity, residuals by score level |
| Not done | Audit of recorded StablePFN numbers against the actual test set |

StablePFN was **not** retrained. That is teammate-owned work.

## 2. Methodology

**Target.** `Addicted_Score`, confirmed in the CSV as an integer from 2 to 9 (705 rows). `Student_ID` is excluded from the features.

**Split protocol.**
1. Re-created the original split exactly: `train_test_split(test_size=0.2, random_state=42)`, then 12.5% of the remainder as validation, giving 493 / 71 / 141 rows. An assertion checks these sizes.
2. The original split is random, seeded and reproducible, so it is valid to reuse. The **141-row test set is kept as-is**, which makes the baselines directly comparable with the recorded StablePFN test score.
3. Train and validation are merged into a **564-row development set**. A single 71-row validation set is too small for reliable model selection.
4. The test row indices are saved to `results/test_indices.csv`.

**Preprocessing.** A scikit-learn `ColumnTransformer` inside a `Pipeline`, so it is re-fitted on training rows only in every fold.

| Feature(s) | Treatment |
|---|---|
| Numeric columns | `StandardScaler` |
| Binary columns | 0/1 |
| Academic_Level | Ordinal encoding |
| Relationship_Status, Platform | One-hot |
| Country | One-hot, with countries seen fewer than 10 times grouped as "infrequent" (110 countries, 79 singletons) |

**Models.**
- DummyRegressor (predicts the mean)
- LinearRegression
- Ridge (alpha tuned)
- RandomForest (depth and leaf size tuned)
- XGBoost (learning rate, depth and number of trees tuned)

**Model selection.** Nested cross-validation.
- Outer loop: 5-fold × 3 repeats = 15 folds, identical for every model.
- Inner loop: 5-fold `GridSearchCV`.
- Each final model is tuned on the full development set and scored **once** on the test set.
- Test-set uncertainty: 2,000-sample bootstrap 95% confidence intervals.

**Metrics.** MAE, MSE, RMSE, R², plus the share of predictions within ±1 point (descriptive). Metrics were recomputed by hand and matched scikit-learn.

## 3. Results (VERIFIED)

### 3.1 Model comparison (`results/model_comparison.csv`)

| Model | CV RMSE (mean ± sd, 15 folds) | CV R² | Test MAE | Test RMSE [95% CI] | Test R² [95% CI] |
|---|---|---|---|---|---|
| Dummy (mean) | 1.588 ± 0.065 | −0.009 | 1.378 | 1.584 [1.446, 1.728] | −0.003 |
| LinearRegression | 0.278 ± 0.025 | 0.969 | 0.184 | 0.252 [0.208, 0.299] | 0.975 [0.962, 0.983] |
| Ridge (alpha = 1) | 0.276 ± 0.025 | 0.969 | 0.184 | 0.249 [0.206, 0.293] | 0.975 [0.963, 0.983] |
| RandomForest | 0.170 ± 0.026 | 0.988 | 0.038 | 0.149 [0.064, 0.217] | 0.991 [0.982, 0.998] |
| **XGBoost** (best by CV) | **0.164 ± 0.018** | **0.989** | 0.069 | 0.178 [0.092, 0.252] | 0.987 [0.975, 0.997] |

How to read this:
- **Tree ensembles beat linear models clearly.** XGBoost had lower RMSE than both linear models in all 15 folds.
- **XGBoost and RandomForest are statistically indistinguishable.** XGBoost wins on CV (12 of 15 folds, by only 0.0065 RMSE on average), RandomForest wins on test, and their confidence intervals overlap heavily. Neither should be called "the best" model.
- **CV agrees with test.** Every test RMSE falls within the range of that model's 15 CV fold scores (`fig02_cv_vs_test.png`), so the test numbers are not a lucky split.
- **Tuning was stable.** XGBoost picked the same settings in 12 of 15 outer folds.

### 3.2 Reliability checks

**Overfitting.**
- Gap between validation and training RMSE: RandomForest 0.098, XGBoost 0.111, Linear 0.030.
- The trees fit their training rows much more tightly, but their held-out error is still the lowest.
- Learning curves (`fig03`) show the CV error still falling at 450 rows.

**Split sensitivity.** I re-split the development set 50 times (80/20, test set untouched).
- XGBoost R² ranged from 0.976 to 0.994, and RMSE from 0.123 to 0.249.
- So a single-split RMSE can move by about ±0.06 purely by chance (`fig06`, `split_sensitivity.csv`).

**Residuals** (`fig05`, `residuals_by_score_level.csv`).
- Residuals centre on 0 (mean −0.028).
- The largest error (1.29) is the **only score-2 student in the entire dataset**. That student is in the test set, so no model ever trained on a score of 2.
- Rare extreme scores are a known blind spot.

### 3.3 Are the very high scores caused by leakage?

No evidence of leakage was found.

| Check | Result |
|---|---|
| Duplicates | 0 exact duplicates. 1 pair is identical apart from `Student_ID` (IDs 242 and 274, same target), and both are in the development set, so nothing crosses into the test set. 0 identical feature vectors with different targets. |
| ID leakage | Spearman correlation between `Student_ID` and the target is 0.032. The ID is excluded anyway. |
| Shuffled-target test (100 permutations, `fig07`) | Real-target CV R² = 0.990. Shuffled-target R²: mean −0.237, max −0.132, p ≈ 0.01 (the smallest p-value possible with 100 shuffles). The pipeline learns nothing from noise, so it does not leak. |
| Metric calculation | Hand-computed MAE, MSE, RMSE and R² equal scikit-learn's. |

**Why the scores are so high:** the target is close to a function of a few survey answers.

Single-feature CV R² (`single_feature_r2.csv`):

| Feature alone | CV R² |
|---|---|
| Mental_Health_Score | 0.892 |
| Conflicts_Over_Social_Media | 0.863 |
| Affects_Academic_Performance | 0.743 |
| Avg_Daily_Usage_Hours | 0.690 |

Feature-set sensitivity (`feature_set_sensitivity.csv`, `fig08`), CV R²:

| Feature set | XGBoost | RandomForest | Ridge |
|---|---|---|---|
| Without the outcome-like features (mental health, conflicts, academic impact) | 0.943 | 0.933 | 0.854 |
| Only Mental_Health_Score + Conflicts | 0.949 | 0.953 | 0.929 |

R² above 0.93 from usage, sleep and demographics alone is unusually clean for real survey data. **This points to a synthetic or rule-generated dataset rather than a leak.**
High accuracy here does not show that the models would work on real students.

**Country encoding** (`country_encoding_comparison.csv`). The original method (unsmoothed mean target per country), fitted correctly inside CV, gave RandomForest a CV RMSE of 0.211 ± 0.050:
- worse than rare-grouped one-hot: 0.171 ± 0.025
- worse than scikit-learn's cross-fitted `TargetEncoder`: 0.189 ± 0.030
- with the largest overfitting gap (0.142)

The original encoding is not a test-set leak. It is a noisy, overfitting-prone choice.

### 3.4 Feature importance (descriptive, not causal)

Permutation importance on the test set (`fig09`, `permutation_importance_test.csv`) ranks the features in the same order for all three models:
1. `Mental_Health_Score`
2. `Conflicts_Over_Social_Media`
3. `Affects_Academic_Performance` / `Country`
4. sleep and usage hours

Gender, Age and Relationship_Status contribute almost nothing.
These numbers show **what the models rely on**. They are not causal effects. Predicted scores are not clinical diagnoses.

## 4. Audit of previously recorded results

Files: `previous_results_audit.csv`, `original_pipeline_reproduction.csv`.

| Recorded claim (StablePFN notebook) | Check | Status |
|---|---|---|
| Test MSE 0.8935, MAE 0.7146, R² 0.6429 | Test-set variance is 2.5023, so the recorded MSE implies R² = 0.6429 | **Internally consistent** with the 141-row test set. The model itself was not re-run, so the result stays **RECORDED (unverified)**. |
| Binary task "≥ 6.0", confusion matrix TN 62 / FN 79 / TP 0 | The test set has 79 rows with score ≥ 7 and 88 with score ≥ 6 | **VERIFIED inconsistency:** the code used threshold **7.0**, so the "≥ 6.0" label is wrong. Precision, recall and F1 are 0 because the model never predicted ≥ 7. |
| Hard gate masking at 0.5 → MSE 22.93 | Mean-only Dummy test MSE is 2.51 | Plausible. All gates are below 0.05, so masking zeroes every input and the model does far worse than predicting the mean. |
| Run-2 test metrics; iLTM, TabSRA, multiclass, NLP and other baseline results | Not readable or not provided | **UNVERIFIED** |

**Original preprocessing re-created** (unscaled inputs, naive Country encoding, `get_dummies`, 22 features, the original 493-row training split). Test-set results:

| Model | Test R² | Test RMSE |
|---|---|---|
| LinearRegression | 0.971 | 0.268 |
| RandomForest | 0.989 | 0.166 |
| XGBoost | 0.984 | 0.201 |
| StablePFN (RECORDED) | 0.643 | 0.945 |

Under the *same* split and preprocessing, a plain linear regression explains about 97% of the variance, compared with 64% for the recorded StablePFN.
So StablePFN's weaker score most likely comes from its **model or training design**, not from the data or the preprocessing. That fits the audit findings: collapsed gates, learned sample weights in the loss, and checkpoint selection on the composite loss. This is an inference from the evidence; StablePFN was not re-run to confirm it.

## 5. Limitations

- **The dataset appears synthetic or rule-generated.** The results describe this dataset, not real-world addiction. The redistribution licence is unclear, so the CSV is not committed (place it in `data/` locally).
- **The sample is small** (705 rows), so test confidence intervals are wide relative to the differences between the tree models.
- **Score 2 appears once**, and only in the test set. The models cannot learn the lowest score.
- **StablePFN was not re-run.** Its numbers are checked only for internal consistency.
- **Only about 21% of the StablePFN notebook export was readable** (embedded images exceed the reader limit). Cells after "## 11. Plots" were not audited. **The friend's notebook was not available** in this workspace, so its results could not be compared.
- Permutation importance on correlated features can spread or hide importance between them. It is not causal.
- Some reliability checks use the test set after model selection: test-set permutation importance, and the original-preprocessing re-creation. These are diagnostics only. No model or setting was chosen from them.

## 6. Reproducing

```bash
pip install -r requirements-reliability.txt
# put the CSV at data/Students-Social-Media-Addiction-1.csv (or set SMA_DATA_PATH)
jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_reliability_analysis.ipynb
```

Runtime is about 5 minutes on 2 CPU cores. All outputs are written to `results/` and `results/figures/`.

## 7. Suggested next steps for the team

1. Fix the binary-threshold label in the StablePFN notebook (7.0 vs 6.0).
2. Select StablePFN checkpoints on validation MSE, not the composite loss.
3. Report StablePFN with the same protocol as here: nested CV plus the fixed test set.
4. Treat the gate scores as non-causal. Do not interpret them while all gates sit near 0.02.
