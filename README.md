# Student Social Media Addiction: Dataset Inspection and EDA

**Course:** CSE445  
**Contributor:** Md Sabbir Hossain Tamim

## Objective

Explore the student social-media addiction dataset, assess its quality, and document patterns before preprocessing and model training.

This README describes the work completed in `notebooks/01_data_inspection.ipynb`. Model training and evaluation are handled separately by another team member.

## Dataset

- **File:** `Students-Social-Media-Addiction-1.csv`
- **Size:** 705 student records and 13 columns
- **Data types:** 5 integer, 2 floating-point, and 6 text columns
- **Target for subsequent modeling:** `Addicted_Score`

| Column | Content |
| --- | --- |
| `Student_ID` | Student identifier |
| `Age` | Student age |
| `Gender` | Gender category |
| `Academic_Level` | Academic level |
| `Country` | Country label |
| `Avg_Daily_Usage_Hours` | Average daily social-media usage in hours |
| `Most_Used_Platform` | Most-used social-media platform |
| `Affects_Academic_Performance` | Yes/No response about academic impact |
| `Sleep_Hours_Per_Night` | Nightly sleep duration in hours |
| `Mental_Health_Score` | Recorded mental health score |
| `Relationship_Status` | Relationship category |
| `Conflicts_Over_Social_Media` | Recorded social-media conflict value |
| `Addicted_Score` | Recorded addiction score |

The dataset source, collection method, scoring definitions, and target construction still need to be documented from the original source. Observed score ranges should not be treated as documented scale limits.

## Project Layout

The notebook expects this layout:

```text
CSE445_ML_project/
|-- data/
|   `-- Students-Social-Media-Addiction-1.csv
|-- notebooks/
|   `-- 01_data_inspection.ipynb
`-- README.md
```

## How to Run

1. Open the repository folder in VS Code.
2. Install Python and the VS Code Python and Jupyter extensions if needed.
3. Install the required packages:

   ```bash
   python -m pip install pandas matplotlib ipykernel
   ```

4. Place the CSV in the `data` folder using the filename shown above.
5. Open `notebooks/01_data_inspection.ipynb` and select the Python kernel containing these packages.
6. Run all cells from top to bottom. The notebook resolves the dataset path when the working directory is the project root or its `notebooks` folder.

Saved cell outputs may remain visible after restarting Python; rerun the loading cell to recreate `df` before continuing the analysis.

## Completed Work

1. Loaded the dataset and previewed sample records.
2. Inspected column names, data types, and non-null counts.
3. Checked missing values, blank text, duplicate records, and repeated identifiers.
4. Inspected categorical labels and surrounding whitespace.
5. Reviewed numerical summary statistics and category counts.
6. Created an analysis copy with trimmed categorical values.
7. Visualized addiction scores, daily usage, and platform counts.
8. Examined daily usage against addiction score using a scatter plot.
9. Created a numerical correlation heatmap excluding `Student_ID`.
10. Inspected potential outliers using boxplots and the 1.5 × IQR rule.

## Data Quality Findings

| Check | Result |
| --- | --- |
| Missing values detected by `isna()` | 0 |
| Blank categorical entries after stripping spaces | 0 |
| Exact duplicate rows | 0 |
| Repeated student IDs | 0 |
| Duplicate occurrences excluding `Student_ID` | 1 |
| Categorical entries with surrounding whitespace | 705 in each of 6 columns |

Students **242 and 274** have identical values across the other 12 columns. Both records were retained because matching responses alone do not establish an error.

Column names were trimmed in `df`. A separate copy, `df_eda`, was created for exploration, and leading and trailing spaces were removed from its categorical values. Internal spaces in labels were preserved. No records were removed.

## Descriptive Statistics

| Variable | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Age | 20.66 | 21.00 | 18.00 | 24.00 |
| Daily usage, hours | 4.92 | 4.80 | 1.50 | 8.50 |
| Sleep per night, hours | 6.87 | 6.90 | 3.80 | 9.60 |
| Mental health score | 6.23 | 6.00 | 4.00 | 9.00 |
| Social-media conflicts | 2.85 | 3.00 | 0.00 | 5.00 |
| Addiction score | 6.44 | 7.00 | 2.00 | 9.00 |

The most common addiction score is **7**.

## Exploratory Findings

### Sample Composition

- Gender counts are nearly balanced: **353 female** and **352 male** students.
- Academic levels include **353 undergraduates**, **325 graduates**, and **27 high school students**.
- Relationship categories include **384 single**, **289 in a relationship**, and **32 complicated**.
- Country representation is uneven, including countries represented by only one student.

### Platform Distribution

| Platform | Students |
| --- | ---: |
| Instagram | 249 |
| TikTok | 154 |
| Facebook | 123 |
| WhatsApp | 54 |
| Twitter | 30 |
| LinkedIn | 21 |
| WeChat | 15 |
| Snapchat | 13 |
| LINE | 12 |
| KakaoTalk | 12 |
| VKontakte | 12 |
| YouTube | 10 |

Instagram, TikTok, and Facebook together represent **74.6%** of the sample, calculated from their combined counts. These figures describe this dataset, not all students.

### Correlations with Addiction Score

| Variable | Pearson correlation |
| --- | ---: |
| Mental health score | -0.945 |
| Social-media conflicts | +0.934 |
| Daily usage hours | +0.832 |
| Sleep hours per night | -0.765 |
| Age | -0.166 |

Higher addiction scores are associated with lower mental health scores, more conflicts, greater daily usage, and fewer sleep hours. Age has a weak negative linear association. These relationships do not establish causation.

### Potential Outliers

For daily usage, Q1 is **4.10 hours**, Q3 is **5.80 hours**, and IQR is **1.70 hours**. The 1.5 × IQR boundaries are **1.55–8.35 hours**.

| Student ID | Daily usage, hours | Addiction score |
| --- | ---: | ---: |
| 7 | 1.5 | 2 |
| 487 | 8.4 | 9 |
| 497 | 8.5 | 9 |

These **3 records (0.43%)** were retained because statistical outliers are not necessarily data errors. The other five plotted numerical variables showed no outliers under this boxplot rule.

## Limitations and Modeling Handoff

- Representativeness of the wider student population has not been established.
- Small and unequal country, platform, and academic groups limit comparisons.
- Missing-value and duplicate checks do not establish that responses are accurate.
- The source documentation should be checked to determine how the target was constructed and whether any predictors contributed to it. Strong correlations alone do not prove leakage.
- The notebook creates an analysis copy in memory; it does **not** export a cleaned CSV or train models.
- The modeling team should explicitly select its input file, apply consistent whitespace cleaning, exclude `Student_ID` from model inputs, and fit learned preprocessing within the training and cross-validation pipeline.

## Outputs

The notebook contains inspection results, written findings, a score-distribution bar chart, a daily-usage histogram, a platform-count chart, a usage-versus-score scatter plot, a correlation heatmap, and six numerical boxplots.
