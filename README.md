# Credit Risk Modeling – Probability of Default (PD)

## Overview
This project demonstrates an end-to-end credit risk analytics workflow using Python.
The objective is to estimate Probability of Default (PD) for retail loan customers using a logistic regression model.

## Dataset
A synthetic loan dataset was created with the following features:
- Age
- Income
- Loan Amount
- Credit Score
- Loan Tenure
- Past Defaults

A binary target variable (Default) was generated using risk-based rules.

## Methodology
1. Data generation and preprocessing
2. Exploratory Data Analysis (EDA)
3. Credit risk segmentation using credit score bands
4. Logistic regression model for PD estimation
5. Model evaluation using classification metrics and ROC-AUC
6. Feature importance analysis using model coefficients

## Key Results
- Calculated portfolio-level default rate
- Observed monotonic decrease in default risk with improving credit score
- Identified key risk drivers such as credit score, past defaults, and loan amount
- Achieved strong model discrimination measured by ROC-AUC

## Tools & Technologies
- Python
- Pandas
- NumPy
- Scikit-learn

## Business Interpretation
Higher credit scores and income reduce default probability, while higher loan amounts and past defaults increase risk, consistent with credit risk theory.
