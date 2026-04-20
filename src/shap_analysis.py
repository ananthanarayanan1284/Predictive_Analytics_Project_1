"""
SHAP Analysis Module for Telecom Customer Churn Prediction
==========================================================
Computes SHAP values for model explainability and generates retention insights.
"""

import shap
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

def compute_shap_values(model, X_test, feature_names):
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_df)
    except:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
        shap_values = explainer.shap_values(X_test_df)
    return shap_values, X_test_df

def compute_and_save_shap(model, X_test, feature_names, save_path='models/shap_values.pkl'):
    shap_values, X_test_df = compute_shap_values(model, X_test, feature_names)
    joblib.dump({'values': shap_values, 'data': X_test_df}, save_path)
    return shap_values, X_test_df

def get_top_churn_drivers(shap_values, feature_names, n=10):
    shap_matrix = shap_values.values
    if len(shap_matrix.shape) == 3:
        shap_matrix = shap_matrix[:, :, 1]
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n]
    drivers = []
    for idx in top_indices:
        feature = feature_names[idx]
        importance = mean_abs_shap[idx]
        impact = "↑ Churn" if shap_matrix[:, idx].mean() > 0 else "↓ Retention"
        drivers.append({'feature': feature, 'importance': importance, 'impact': impact})
    return pd.DataFrame(drivers)

def generate_retention_insights(shap_values, feature_names, data=None):
    drivers = get_top_churn_drivers(shap_values, feature_names, n=15)
    insights = []
    for _, row in drivers.iterrows():
        feat = row['feature']
        if 'Contract_Two year' in feat:
            insights.append({
                "title": "Long-term Security", 
                "description": "High importance of long-term contracts in preventing churn.", 
                "action": "Incentivize month-to-month customers to switch to 2-year contracts.", 
                "impact": "HIGH"
            })
        elif 'tenure' in feat:
            insights.append({
                "title": "Onboarding Focus", 
                "description": "Short tenure is a major driver of customer departure.", 
                "action": "Launch a 90-day onboarding program for new customers.", 
                "impact": "HIGH"
            })
        elif 'Fiber optic' in feat:
            insights.append({
                "title": "Fiber Quality Audit", 
                "description": "Fiber optic service correlates with higher dissatisfaction.", 
                "action": "Investigate service quality issues for fiber optic users.", 
                "impact": "HIGH"
            })
        elif 'MonthlyCharges' in feat:
            insights.append({
                "title": "Value Optimization", 
                "description": "High monthly charges increase the likelihood of churn.", 
                "action": "Review pricing tiers for high-paying customers.", 
                "impact": "MEDIUM"
            })
        elif 'Electronic check' in feat:
            insights.append({
                "title": "Auto-Pay Migration", 
                "description": "Electronic check users churn more than auto-pay users.", 
                "action": "Offer small discounts for switching to Credit Card/Bank auto-pay.", 
                "impact": "MEDIUM"
            })
    return insights
