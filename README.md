# Fraud Detection System  

## Overview  

This repository contains an exploratory analysis and implementation of a fraud detection system for a webshop. The project addresses the challenge of detecting fraud patterns that often require contextual analysis of multiple orders, as outlined in the following Problem Statement.
The solution combines both rule-based and machine learning approaches to identify fraudulent orders in real-time. 

## Problem Statement  

Fraudulent orders in the webshop can only be identified effectively by analyzing patterns across multiple orders. The goal is to build a performant algorithm that:  
- Accepts or rejects orders in *real-time*.  
- Maximizes profit by minimizing fraudulent transactions while maximizing legitimate ones.  
- Evaluates performance based on profit uplift compared to a baseline (no fraud detection).  

### Key Metrics:  
- **Profit Calculation**:  
  - Accepted & Legitimate: +1% of order price.  
  - Accepted & Fraudulent: -100% of order price.  
  - Rejected: No profit or loss.  

## Solution Overview 

This repository presents a solution combining **rule-based methods** derived from EDA and a **LightGBM-based machine learning model** for real-time fraud detection.

### Key Features:
- Extracted temporal and behavioral features such as order frequency, price variance, and device usage
- Built and evaluated a LightGBM model with real-time decision capability
- Compared model-based decisions with baseline and rule-based methods using profit uplift as a key metric



## Repository Structure

- **`Exploratory Data Analysis_summary.ipynb` / `.html`**  
  - Analyzes transaction patterns
  - Identifies fraud signatures and builds basic rule-based logic
  - Benchmarks profit uplift using heuristic rules

- **`fraud_detector.py`**  
  - Contains the real-time fraud detection model using LightGBM
  - Uses carefully engineered features to enhance fraud prediction

- **`demo.py`**  
  - Demonstrates how to run the model on incoming data
  - Prints fraud probabilities, confidence scores, and classification outputs
  - Generates evaluation metrics, including AUC-ROC, Average Precision, and profit uplift


- **`requirements.txt`**  
  - Lists all necessary Python libraries

## Usage  

### Installation  
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo:  
   ```bash
   python demo.py
   ```

### Sample Output  
```bash
Loading cached features...
Performing cross-validation...
Training until validation scores dont improve for 50 rounds
Did not meet early stopping. Best iteration is:
[299]   cv_aggs valid auc: 0.980724 + 0.00283477       cv_aggs valid average_precision: 0.674361 + 0.0170146
Best CV AUC: 0.9807 ± 0.0028
Best CV AP: 0.6744 ± 0.0170

Training final model...
Training until validation scores dont improve for 50 rounds
Did not meet early stopping. Best iteration is:
[299]   valid_0 auc: 0.981739 valid_0 average_precision: 0.697303

Model Evaluation:
AUC-ROC Score: 0.9817
Average Precision Score: 0.6973
'AUC-PR Score': 0.6972
Optimal Threshold: 0.5444
'Best Validation Profit: $147250.21'
Baseline Profit (No ML Model): $68333.69

'Classification Report':
              precision    'recall' f1-score   support

       False       1.00      0.97      0.99    103197
        True       0.17      0.88      0.29       633

    accuracy                           0.97    103830
   macro avg       0.59      0.92      0.64    103830
weighted avg       0.99      0.97      0.98    103830

=== Real-Time Fraud Detection ===
Is Fraudulent: False
Fraud Probability: 0.1735
Confidence Score: 0.65
```

---
