
# Problem Statement and Objectives

## 1. Problem Statement

Social media has become an important part of students' daily lives. However, patterns of social media usage may be associated with sleep habits, mental well-being, and daily activities. Understanding these patterns can help researchers explore students' self-reported social media addiction levels.

The main problem addressed by this project is how to predict a student's self-reported social media addiction score using survey information. The project uses the `Addicted_Score` variable, which ranges from 2 to 9, as the target for prediction. Relevant survey features include social media usage, sleep, mental health, and background information.

The project aims to develop and evaluate a machine learning model that can estimate this score from the available survey data. The predictions represent survey-scale estimates and should not be interpreted as clinical diagnoses or proof of causal relationships.

## 2. Project Objectives

The main objectives of this project are:

1. To explore the dataset and understand the relationships and patterns among the available survey variables.
2. To identify relevant features related to social media usage, sleep, mental health, and student background.
3. To develop a transformer-based PyTorch model for predicting self-reported social media addiction scores.
4. To compare the proposed model with baseline regression models.
5. To evaluate model performance, reliability, and limitations using appropriate metrics and experiments.
6. To investigate whether additional classification and natural language processing experiments provide useful insights.
7. To communicate the findings responsibly, without treating model predictions as clinical diagnoses or causal evidence.

## 3. Expected Outcome

The expected outcome is a machine learning model that estimates students' self-reported social media addiction scores from survey responses. The project will also provide an evaluation of model performance, comparisons with baseline methods, and an assessment of the model's reliability and limitations.

## 4. Scope and Limitations

This project focuses on prediction using the available survey data. The model's results depend on data quality, feature selection, and evaluation methods. Any identified feature importance indicates which variables the model relies on; it does not establish that those variables cause social media addiction.
