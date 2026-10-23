# Student Social Media Addiction Score Prediction

Predicting a student's self-reported **social media addiction score** (`Addicted_Score`, 2–9) from survey answers about usage, sleep, mental health and background. The project combines:

- exploratory data analysis
- a custom transformer-based PyTorch model 
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


