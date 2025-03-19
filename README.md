# Credit Risk Modeling Project

This repository contains an ongoing project to predict loan defaults using machine learning techniques. The goal is to build a model that accurately identifies the likelihood of a borrower defaulting on a loan, enabling better decision-making for loan approvals and risk management.

## Project Overview

- **Objective**: Predict the probability of loan default (PD) using historical loan data.
- **Dataset**: Lending Club Loan Data (`accepted_2007_to_2018Q4.csv`) with 2,260,701 records and 151 features.
- **Focus Metrics**: AUC-ROC, Gini Coefficient, KS Statistic, Precision, Recall, F1-Score.
- **Status**: Currently in the modeling phase, with baseline logistic regression and advanced XGBoost models under development.

## Setup

### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `xgboost`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Environment
- Developed in Jupyter Notebook.
- Version control managed with Git.

## Project Structure
- `data/`: Contains the raw dataset (not uploaded due to size; sourced from Kaggle).
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, and modeling.
- `src/`: Python scripts for reusable code (to be added).
- `README.md`: This file.

## Progress

### 1. Data Acquisition
- Loads the Lending Club dataset (accepted_2007_to_2018Q4.csv) with 2.26M rows and 151 columns.
- Initial shape: (2260701, 151).

### 2. Data Cleaning
- Drops columns with >90% missing values (38 columns removed), reducing shape to (2260701, 113).
- Removes duplicates (none found).
- Fixes inconsistencies:
  - Negative dti values set to 0.
  - Imputes missing dti with median.
  - Verifies no negative values in key numeric columns (loan_amnt, annual_inc, fico_range_low, int_rate).

### 3. Exploratory Data Analysis (EDA)
- Defines default as a binary target (1 for 'Charged Off'/'Default', 0 for 'Fully Paid', NaN for others).
- Filters dataset to finalized loan statuses, resulting in shape (1345350, 114) with a default rate of ~19.96%.
- Visualizes key numeric features (loan_amnt, annual_inc, dti, fico_range_low, int_rate) with histograms and boxplots vs. default status.
- Applies log-transformation to annual_inc to address skewness, creating annual_inc_log.
- Confirms FICO scores range from 625 to 845 (within expected domain).
- Generates a correlation heatmap for key features and default.

### 4. Data Preprocessing
- **Encoding**: One-hot encodes categorical variables (grade, sub_grade, home_ownership, verification_status, purpose, term).
- **Scaling**: Standardizes numeric features (loan_amnt, annual_inc_log, dti, fico_range_low, int_rate, installment, revol_bal) using StandardScaler.
- **Feature Engineering**:
  - Creates loan_to_income and credit_util ratios, scaled with StandardScaler.
- **Class Imbalance**: Applies SMOTE to balance training data, achieving a 50% default rate in the balanced set ((1507452, 136)).
- **Cleanup**: Drops leakage-prone columns (e.g., total_pymnt_inv, recoveries) and high-missingness columns; imputes remaining NaNs with medians.

### 5. Modeling
- **Train-Test Split**: 70-30 split, stratified by default, yielding training shape (941745, 160) and test shape (403605, 160).
- **Baseline Model (Logistic Regression)**:
  - Initial run with leakage showed AUC ~0.998, revealing data leakage.
  - After removing leakage columns, AUC drops to ~0.632, Gini ~0.263, KS ~0.191.
  - Classification report shows balanced precision/recall trade-offs post-leakage fix.
- **Advanced Model (XGBoost)**:
  - Trains with SMOTE-balanced data and eval_metric='auc'.
  - Encounters feature mismatch issues (ongoing debugging).

## Current Results

### Logistic Regression (Post-Leakage Fix)
- **AUC**: 0.632
- **Gini**: 0.263
- **KS Statistic**: 0.191
- **Classification Report**:
  ```
              precision    recall  f1-score   support
         0.0       0.86      0.55      0.67    323025
         1.0       0.26      0.64      0.37     80580
  accuracy                            0.57    403605
  ```

## Visualizations
- ROC Curve (post-leakage fix) plotted with AUC annotated.

## Next Steps
- Resolve XGBoost feature mismatch error and evaluate its performance.
- Experiment with additional models (e.g., Random Forest, LightGBM).
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Enhance feature engineering with domain-specific features (e.g., payment history trends).
- Develop a deployment plan (e.g., Flask API or cloud endpoint).
- Create a business-focused dashboard with risk segmentation insights.

## Limitations
- Current model performance is modest post-leakage fix (~0.63 AUC), indicating room for feature or model improvement.
- High-cardinality categorical variables (e.g., emp_title) dropped; potential for target encoding.
- Assumes static data; no time-series validation yet.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in notebooks/ and run cells sequentially.

## Contributing
Feel free to fork this repo, submit issues, or contribute pull requests as I refine this project!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
