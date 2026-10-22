# Student Social Media Addiction Score Prediction

Predicting a student's self-reported **social media addiction score** (`Addicted_Score`, 2–9) from survey answers about usage, sleep, mental health and background. The project combines:

- exploratory data analysis
- a custom transformer-based PyTorch model (**StablePFN**)
- baseline regression models
- a model reliability audit
- secondary classification and NLP experiments

> **Important:** a predicted score is a number on a survey scale. It is **not** a clinical diagnosis.
> Feature importance and gate scores show what a model relies on. They are **not** causal effects.

---

## Table of contents
1. [Dataset](#dataset)
2. [Repository structure](#repository-structure)
3. [Project components](#project-components)
4. [Results summary](#results-summary)
5. [Setup and running](#setup-and-running)
6. [Reproducibility](#reproducibility)
7. [Known issues and limitations](#known-issues-and-limitations)
8. [Team contributions](#team-contributions)

---

## Dataset

**File:** `Students-Social-Media-Addiction-1.csv`. It has 705 rows and 13 columns, with no missing values.

| Column | Type | Notes |
|---|---|---|
| `Student_ID` | integer | Identifier only, never used as a feature |
| `Age` | integer | 18–24 |
| `Gender` | category | Female / Male |
| `Academic_Level` | category | High School / Undergraduate / Graduate |
| `Country` | category | 110 countries, 79 of them appear only once |
| `Avg_Daily_Usage_Hours` | float | 1.5–8.5 |
| `Most_Used_Platform` | category | 12 platforms (Instagram, TikTok, Facebook, …) |
| `Affects_Academic_Performance` | category | Yes / No |
| `Sleep_Hours_Per_Night` | float | 3.8–9.6 |
| `Mental_Health_Score` | integer | 4–9 |
| `Relationship_Status` | category | Single / In Relationship / Complicated |
| `Conflicts_Over_Social_Media` | integer | 0–5 |
| **`Addicted_Score`** | integer | **Target, 2–9** (mean 6.44). Score 2 appears only once. |

**The dataset is not included in this repository**, because its redistribution permission is unclear. Place your copy at:

```
data/Students-Social-Media-Addiction-1.csv
```

Make sure `data/` is listed in `.gitignore`, so the dataset is never committed.

---

## Repository structure

```
.
├── README.md
├── requirements-reliability.txt          # packages for the reliability notebook
├── data/                                 # NOT committed: place the CSV here
├── Dataset_Plotting_Preprocessing.ipynb  # EDA and exploratory preprocessing
├── Student_Social_Media_Addiction_Score_StableFPN.ipynb   # StablePFN model (Colab)
├── notebooks/
│   └── 03_model_reliability_analysis.ipynb   # fair evaluation and reliability audit
├── results/
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   ├── previous_results_audit.csv
│   ├── original_pipeline_reproduction.csv
│   ├── split_sensitivity.csv, feature_set_sensitivity.csv, ...
│   └── figures/                          # fig01 … fig09 (.png)
└── docs/
    └── initial_experiment.md             # reliability analysis write-up
```

> Notebook file names above match the versions shared with the team. Adjust them if your repository uses different names.
> <!-- TODO: add the teammate notebook(s) for iLTM, TabSRA, classification and NLP experiments here. -->

---

## Project components

### 1. Exploratory data analysis: `Dataset_Plotting_Preprocessing.ipynb`
- Inspects the data: shape, data types, duplicates (none) and missing values (none).
- Plots: scatter of usage vs score, KDE of the score, histograms, box, swarm, strip and violin plots.
- Counts outliers with the IQR rule. It finds 3 in `Avg_Daily_Usage_Hours`.
- Demonstrates `StandardScaler` and `LabelEncoder` on the full dataset. This is **for exploration only**: these transformed tables are not used for modelling.

### 2. StablePFN model: `Student_Social_Media_Addiction_Score_StableFPN.ipynb`
A custom PyTorch model for tabular regression, inspired by transformer-based tabular models. It runs in Google Colab with a GPU.

| Part | Details |
|---|---|
| Split | `train_test_split(test_size=0.2, random_state=42)`, then 12.5% of the remainder as validation, giving **493 / 71 / 141** rows |
| Preprocessing | Binary and ordinal mappings, Country replaced by its mean training-set target, one-hot Platform. 22 features. |
| Architecture | Each feature becomes a 128-dimensional token plus a feature-ID embedding. Sigmoid **feature gates**. 2-layer transformer encoder (4 heads). Prediction head, plus a "decoupling" head that learns per-sample weights. 342,787 parameters. |
| Loss | Weighted prediction error + weight entropy + gate-polarisation, sparsity and logit-L2 terms |
| Training | AdamW optimiser (learning rate 1e-4), cosine learning-rate schedule, gate-loss warm-up, early stopping (patience 15) |

### 3. Model reliability analysis: `notebooks/03_model_reliability_analysis.ipynb`
Evaluates regression models fairly and checks whether the reported performance can be trusted. Full write-up: [`docs/initial_experiment.md`](docs/initial_experiment.md).

- **Same test set:** reuses the original split, so all models are judged on the **same 141 test rows** as StablePFN.
- **Leakage-safe preprocessing:** scikit-learn `Pipeline` + `ColumnTransformer`, re-fitted on training rows only in every fold.
- **Nested cross-validation** on the 564-row development set (original train + validation): 5-fold × 3 repeats outside, with a 5-fold grid search inside for tuning.
- **Models:** Dummy (predicts the mean), LinearRegression, Ridge, RandomForest, XGBoost.
- **Test set used once,** with 95% bootstrap confidence intervals.
- **Reliability checks:** CV vs test, overfitting gap, learning curves, residuals by score level, sensitivity to the split.
- **Leakage checks:** shuffled-target permutation test, single-feature models, Country-encoding comparison, feature-set sensitivity, metrics recomputed by hand.
- **Audit** of the results recorded in the StablePFN notebook.

### 4. Secondary experiments
These are separate from the main regression task:
- binary and multiclass classification
- iLTM and TabSRA comparisons
- NLP experiments on text descriptions generated from the table rows

<!-- TODO: link the notebook(s) and summarise these once they are added to the repository. -->

---

## Results summary

Labels used here:

| Label | Meaning |
|---|---|
| **VERIFIED** | Produced by running `03_model_reliability_analysis.ipynb` (two runs gave identical numbers) |
| **RECORDED** | Taken from a notebook's saved output and not re-run |

### Regression on the 141-row test set

| Model | Status | Test MAE | Test RMSE | Test R² [95% CI] |
|---|---|---|---|---|
| Dummy (mean) | VERIFIED | 1.378 | 1.584 | −0.003 |
| LinearRegression | VERIFIED | 0.184 | 0.252 | 0.975 [0.962, 0.983] |
| Ridge | VERIFIED | 0.184 | 0.249 | 0.975 [0.963, 0.983] |
| RandomForest | VERIFIED | 0.038 | 0.149 | 0.991 [0.982, 0.998] |
| XGBoost (best by CV) | VERIFIED | 0.069 | 0.178 | 0.987 [0.975, 0.997] |
| StablePFN (run 1) | RECORDED | 0.715 | 0.945 | 0.643 |

### Key findings
- **RandomForest and XGBoost cannot be separated.** XGBoost won cross-validation, RandomForest won the test set, and their confidence intervals overlap.
- **No leakage was found.**
  - With the target shuffled, R² never rose above −0.13 (with the real target it is 0.99).
  - The only duplicate pair (identical apart from `Student_ID`) lies entirely in the development set.
- **The high scores come from the dataset itself.**
  - `Mental_Health_Score` alone gives a CV R² of 0.89.
  - Without the mental-health, conflicts and academic-impact columns, XGBoost still reaches a CV R² of 0.94.
  - The data is likely synthetic, so these results should not be generalised to real students.
- **The recorded StablePFN R² (0.643) is internally consistent** with the test set. However, with the same split and preprocessing, plain linear regression reaches 0.971. The gap most likely comes from the model or training design (an inference; StablePFN was not re-run).

---

## Setup and running

### Reliability analysis (local, CPU)
Verified with Python 3.13.16, scikit-learn 1.9.1 and XGBoost 3.4.1. Exact versions are in `results/environment.json`.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-reliability.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_reliability_analysis.ipynb
```

- The run takes about 5 minutes on 2 CPU cores.
- Outputs are written to `results/` and `results/figures/`.
- The notebook works when started from the repository root or from `notebooks/`.
- If the CSV is somewhere else, set the environment variable `SMA_DATA_PATH` to its path.

### StablePFN (Google Colab)
1. Open the notebook in Colab and choose *Runtime → Change runtime type → GPU*.
2. Upload the CSV, and make sure the file name in the code matches the uploaded file.
3. Run the cells in order. The notebook needs `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` and `tqdm`, all pre-installed on Colab.

### EDA notebook
This needs only `pandas`, `numpy`, `matplotlib`, `seaborn` and `scikit-learn`. Run all cells.

---

## Reproducibility

- **Seeds:** the reliability notebook fixes all random seeds (`SEED = 42`), and two full runs gave identical results.
- **Split:** the test row indices are saved to `results/test_indices.csv`.
- **Versions:** package versions are recorded in `results/environment.json`.
- **StablePFN:** the notebook does **not** set seeds for torch, numpy or the DataLoader, so retraining gives different numbers.

---

## Known issues and limitations

- **Binary classification threshold:** the StablePFN notebook labels its binary task "score ≥ 6.0", but the code uses **7.0**. The test set has 79 rows with score ≥ 7, which matches the recorded confusion matrix. Precision, recall and F1 are 0 because the model never predicted a positive.
- **Feature gates collapsed:** all StablePFN gate scores ended around 0.02. Masking at 0.5 therefore removes every feature, and the gate scores should not be read as "causal" or Markov-boundary features.
- **Checkpoint selection:** StablePFN keeps the checkpoint with the lowest *composite* validation loss, not the lowest prediction error.
- **Country encoding:** the original encoding replaces each country with its mean training target, without smoothing. With 79 single-student countries, this overfits.
- **Small and likely synthetic dataset:** 705 rows, and score 2 appears only once (in the test set).
- **Not yet independently verified:** iLTM, TabSRA, multiclass and NLP results.

---

## Team contributions

| Member | Contribution |
|---|---|
| Himel | Model reliability analysis and experimental evaluation (`notebooks/03_model_reliability_analysis.ipynb`, `results/`, `docs/initial_experiment.md`) |
| <!-- TODO --> | Dataset EDA and preprocessing (`Dataset_Plotting_Preprocessing.ipynb`) |
| <!-- TODO --> | StablePFN model and initial benchmarking |
| <!-- TODO --> | iLTM / TabSRA / classification / NLP experiments |

## License

<!-- TODO: choose a license for the code. The dataset is excluded and keeps its original terms. -->
