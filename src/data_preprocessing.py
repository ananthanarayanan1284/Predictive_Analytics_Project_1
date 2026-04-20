"""
Data Preprocessing Module for Telecom Customer Churn Prediction
================================================================
Handles data loading, cleaning, feature engineering, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']
    df['TotalCharges'].fillna(0, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    print(f"Data cleaned. Missing values: {df.isnull().sum().sum()}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['AvgMonthlyCharge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    def count_services(row):
        count = 0
        for col in service_cols:
            val = str(row[col]).strip()
            if val in ('Yes', 'Fiber optic', 'DSL'):
                count += 1
        return count
    df['NumServices'] = df.apply(count_services, axis=1)
    bins = [0, 12, 24, 48, 60, 72]
    labels = ['0-12', '13-24', '25-48', '49-60', '61-72']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    print(f"Feature engineering complete. New shape: {df.shape}")
    return df

def encode_and_scale(df: pd.DataFrame, fit: bool = True, scaler=None):
    df = df.copy()
    y = df['Churn'].map({'Yes': 1, 'No': 0}).values if 'Churn' in df.columns else None
    drop_cols = ['Churn', 'TenureGroup']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        unique_vals = df[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'NumServices']
    num_cols = [c for c in num_cols if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    df = df.astype(float)
    feature_names = df.columns.tolist()
    X = df.values
    print(f"Encoding complete. Feature matrix shape: {X.shape}")
    return X, y, feature_names, scaler

def full_preprocessing_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    df = load_data(filepath)
    df = clean_data(df)
    df_clean = df.copy()
    df = engineer_features(df)
    X, y, feature_names, scaler = encode_and_scale(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names, scaler, df_clean

def preprocess_single_input(input_dict: dict, scaler, training_columns: list) -> np.ndarray:
    df = pd.DataFrame([input_dict])
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['AvgMonthlyCharge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    def count_services(row):
        count = 0
        for col in service_cols:
            if col in row.index:
                val = str(row[col]).strip()
                if val in ('Yes', 'Fiber optic', 'DSL'):
                    count += 1
        return count
    df['NumServices'] = df.apply(count_services, axis=1)
    if 'TenureGroup' in df.columns:
        df.drop('TenureGroup', axis=1, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        unique_vals = df[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'NumServices']
    num_cols = [c for c in num_cols if c in df.columns]
    if scaler:
        df[num_cols] = scaler.transform(df[num_cols])
    return df.values.astype(float)
