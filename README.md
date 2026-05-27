Credit Card Fraud Detection

## Problem
Banks lose millions to fraudulent transactions every year. 
This project builds a machine learning system to detect fraud automatically.

## Why This Is Hard
Only 0.17% of transactions are fraudulent — severe class imbalance 
means standard accuracy is misleading.

## Approach
- Exploratory Data Analysis to identify fraud patterns
- Feature scaling on Amount and Time columns
- Logistic Regression and Random Forest models
- Evaluated using Precision, Recall and ROC-AUC

## Key Findings
- Fraudulent transactions cluster around smaller amounts
- Fraud is more concentrated in the first time period
- Logistic Regression: 92% recall — best for catching all frauds
- Random Forest: 99% precision — best for minimising false alarms

## Tools Used
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## Dataset
Kaggle — Credit Card Fraud Detection by ULB
