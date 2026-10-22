## Model Reliability Analysis and Experimental Evaluation

### Overview

This contribution focuses on evaluating the reliability, reproducibility, and generalization performance of machine-learning models for the **Student Social Media Addiction Score Prediction** project.

The objective is to go beyond reporting a single test score by examining how model performance changes across data splits, cross-validation folds, feature sets, and evaluation settings.

### Objectives

- Evaluate and compare candidate regression models using consistent evaluation procedures.
- Examine performance variability through cross-validation and alternative train-test splits.
- Audit previously reported experimental results for reproducibility and potential evaluation inconsistencies.
- Investigate feature-set sensitivity and the influence of individual features.
- Analyze prediction errors using predicted-versus-actual plots and residual analysis.
- Document experimental findings, limitations, and recommendations for future evaluation.

### Methodology

The reliability analysis is organized around the following experiments:

1. **Model comparison:** Compare candidate models using regression evaluation metrics.
2. **Cross-validation:** Assess performance consistency across multiple validation folds.
3. **Split sensitivity:** Investigate whether model performance changes with different data splits.
4. **Feature analysis:** Examine feature-set sensitivity, permutation importance, and individual-feature predictive performance.
5. **Error analysis:** Study prediction errors and residual patterns to identify potential weaknesses.
6. **Previous-results audit:** Review earlier experimental results and investigate whether they can be reproduced under consistent evaluation settings.

### Project Deliverables

| File or directory | Description |
|---|---|
| `notebooks/03_model_reliability_analysis.ipynb` | Main notebook containing the reliability experiments and analysis |
| `docs/initial_experiment.md` | Written report documenting the experiments, observations, and limitations |
| `results/model_comparison.csv` | Comparative model evaluation results |
| `results/cross_validation_results.csv` | Cross-validation results |
| `results/split_sensitivity.csv` | Results from alternative data splits |
| `results/feature_set_sensitivity.csv` | Feature-set sensitivity analysis |
| `results/permutation_importance_test.csv` | Permutation-based feature importance results |
| `results/previous_results_audit.csv` | Audit of previously reported results |
| `results/figures/` | Visualizations of model performance, prediction errors, and feature analysis |

### Evaluation Metrics

The experiments use regression metrics where applicable:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted scores.
- **MSE (Mean Squared Error):** Measures the average squared prediction error.
- **RMSE (Root Mean Squared Error):** Expresses prediction error on the target's original scale.
- **R² (Coefficient of Determination):** Measures the proportion of target variance explained by a model relative to a baseline.

Results should be interpreted collectively rather than relying on a single metric.

### Reproducibility

The notebook and supporting CSV files provide a record of the experiments and their outputs. To reproduce the analysis, review the notebook's data-loading paths and environment requirements, install the required dependencies, and execute the notebook in the documented order.

See `requirements-reliability.txt` for the dependencies specific to this analysis and `docs/initial_experiment.md` for further experimental details.

### Limitations

Model performance may vary with
