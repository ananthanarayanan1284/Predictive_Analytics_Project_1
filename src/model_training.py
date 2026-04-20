"""
Model Training Module for Telecom Customer Churn Prediction
============================================================
Handles model selection, training, evaluation, and saving artifacts.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import sys

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import full_preprocessing_pipeline

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
    return models

def train_all_models(X_train, X_test, y_train, y_test, use_smote=True):
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    models = get_models()
    results = {}
    metrics_list = []
    
    for name, model in models.items():
        print(f"Training: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        metrics_list.append(metrics)
        results[name] = {'model': model, 'metrics': metrics, 'y_pred': y_pred, 'y_prob': y_prob}
    
    return results, pd.DataFrame(metrics_list)

def save_model_artifacts(results, metrics_df, feature_names, scaler, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Choose best model based on ROC-AUC
    best_model_name = metrics_df.loc[metrics_df['ROC-AUC'].idxmax(), 'Model']
    
    # Sensitivity check: prefer Random Forest if SVM is slightly better on ROC but worse on Recall
    if best_model_name == 'SVM':
        rf_rec = metrics_df[metrics_df['Model'] == 'Random Forest']['Recall'].values[0]
        svm_rec = metrics_df[metrics_df['Model'] == 'SVM']['Recall'].values[0]
        if rf_rec > svm_rec + 0.05:
            best_model_name = 'Random Forest'
            print("Switching from SVM to Random Forest for better Churn detection sensitivity.")
    
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, os.path.join(save_dir, 'best_model.pkl'))
    joblib.dump(results, os.path.join(save_dir, 'all_results.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'preprocessor.pkl'))
    
    metadata = {
        'best_model_name': best_model_name,
        'metrics': metrics_df.to_dict(orient='records'),
        'feature_names': feature_names
    }
    joblib.dump(metadata, os.path.join(save_dir, 'model_metadata.pkl'))
    print(f"Model artifacts saved. Best model: {best_model_name}")

if __name__ == '__main__':
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test, features, scaler, df_clean = full_preprocessing_pipeline(filepath)
    results, metrics_df = train_all_models(X_train, X_test, y_train, y_test)
    save_model_artifacts(results, metrics_df, features, scaler)
    
    # Compute SHAP values automatically
    try:
        from shap_analysis import compute_and_save_shap
        import torch
        compute_and_save_shap(results['XGBoost' if HAS_XGBOOST else 'Random Forest']['model'], X_test, features)
        print("SHAP values computed and saved.")
    except Exception as e:
        print(f"SHAP computation skipped: {e}")
