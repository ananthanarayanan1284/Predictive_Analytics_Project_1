#  Customer Churn Prediction for Telecom Companies

Classify telecom customers as churners or non-churners using usage behavior and demographics.
This end-to-end ML project uses the Telco Customer Churn dataset, applies SMOTE for class imbalance, trains XGBoost and Random Forest models (alongside LR, SVM, LightGBM), and performs **SHAP analysis** for model explainability.
The dashboard outputs actionable customer retention insights backed by model predictions.

## Team Members
- Ananthanarayanan B
- Lanka Priya
- Abhitha Raj S M
  
## Project Overview

Customer churn is one of the biggest challenges in the telecom industry. 
This project builds a Machine Learning pipeline to:

- **Analyze** customer behavior and identify churn drivers
- **Predict** which customers are likely to churn
- **Visualize** insights through an interactive Streamlit dashboard
- **Recommend** retention strategies based on risk scores


  ### Dataset

- **Source:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 columns (demographics, services, account info)
- **Target:** Churn (Yes/No)  ~26.5% churn rate

## Exploratory Data Analysis (EDA)

- **Contract Type**: Month-to-month contracts show significantly higher churn (3x) than yearly contracts.
- **Internet Service**: Fiber optic users have a higher churn rate compared to DSL users.
- **Tenure**: New customers (0-12 months) are at the highest risk of leaving.
- **Payment Method**: Electronic check users demonstrate higher churn correlation.

## Data Challenges

- **Class Imbalance**: Churners represent only a quarter of the dataset, requiring SMOTE oversampling.
- **Categorical Complexity**: Features like "Internet Service" and "Contract" require careful one-hot encoding.
- **Numerical Scaling**: Monthly and Total charges require standardization for models like SVM and Logistic Regression.
  
##  Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted decision trees |
| LightGBM | Fast gradient boosting framework |
| SVM | Support vector machine with RBF kernel |

### Handling Class Imbalance
- **SMOTE** (Synthetic Minority Oversampling Technique) is applied to balance the training set

### Model Explainability
- **SHAP** (SHapley Additive exPlanations) analysis using TreeExplainer for XGBoost/Random Forest
- Global feature importance, individual customer explanations, feature dependence plots
  

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix, ROC Curve, Feature Importance

## Results

The models were evaluated using ROC-AUC, F1-Score, and Recall to ensure the best identification of high-risk customers.

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **Random Forest ⭐** | **~0.76** | **0.72** | **0.62** | **~0.84** |
| Logistic Regression | ~0.74 | 0.80 | 0.62 | ~0.84 |
| Linear SVM | ~0.75 | 0.76 | 0.62 | ~0.82 |
| XGBoost | ~0.77 | 0.58 | 0.58 | ~0.83 |

> **Random Forest selected as best model** due to its superior ROC-AUC and high sensitivity to real-world high-risk profiles.


##  Streamlit Dashboard

The interactive web application includes **6 pages**:

1. **  Overview**  KPI cards, churn distribution, business insights
2. ** Exploratory Analysis**  Interactive charts, correlation heatmaps
3. ** Predict Churn**  Real-time prediction with risk gauge
4. ** Model Performance**  Metrics comparison, ROC curves, confusion matrices
5. **  SHAP Explainability**  Global/local SHAP analysis, beeswarm plots, feature dependence
6. ** Retention Insights**  SHAP-backed business strategies, executive summary, priority playbook

Link: "https://predictiveanalyticsproject1-p7zhzebq2yjo6orjeidt2v.streamlit.app/"
---
<img width="1771" height="576" alt="Screenshot 2026-04-21 005438" src="https://github.com/user-attachments/assets/22f03278-884e-4c20-90ba-79a750819776" />

<img width="1773" height="590" alt="Screenshot 2026-04-21 005553" src="https://github.com/user-attachments/assets/caca262b-8aae-4527-afd4-0c989b20d735" />

<img width="1770" height="599" alt="Screenshot 2026-04-21 005615" src="https://github.com/user-attachments/assets/51b2de3c-8e8e-4918-8853-6c03e25002c1" />

##  Key Findings

- **Month-to-month contracts** have 3x higher churn than yearly contracts
- **Fiber optic customers** churn more than DSL users
- **Electronic check** payment method correlates with higher churn
- **New customers (0-12 months)** are the most vulnerable segment
- Add-on services (security, backup, support) significantly reduce churn

## Data Science Life Cycle Coverage

| Stage | Files / Process |
|---|---|
| Data Understanding | `churn_analysis.ipynb` |
| Preprocessing | `src/data_preprocessing.py` |
| Feature Engineering | `AvgMonthlyCharge`, `NumServices` |
| Modeling | Random Forest, XGBoost (SMOTE applied) |
| Interpretation | SHAP TreeExplainer |
| Deployment | Streamlit Cloud |

##  Technologies

- **Python 3.8+**
- **Pandas, NumPy**  Data manipulation
- **Scikit-learn**  Model training & evaluation
- **XGBoost, LightGBM**  Advanced ML models
- **SHAP**  Model explainability (Shapley values)
- **Imbalanced-learn**  SMOTE oversampling
- **Plotly** Interactive visualizations
- **Streamlit**  Web application framework
- **Statsmodels**  Statistical analysis


##  License

This project is developed as a capstone project for Predictive Analytics coursework (AY 2025-2027)

