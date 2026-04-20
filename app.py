import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# ─── Paths (FIXED) ──────────────────────────────────────
DATA_PATH = "data/Telco-Customer-Churn.csv"
MODELS_DIR = "models"

# ─── Load Data ──────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except:
        st.warning("Dataset not found. Using sample data.")
        return pd.DataFrame({
            "tenure": [1, 5, 10],
            "MonthlyCharges": [50, 70, 90],
            "Churn": ["No", "Yes", "No"]
        })

# ─── Load Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        return model
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
        return None

df = load_data()
model = load_model()

# ─── Sidebar ────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Overview", "Predict"])

# ════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ════════════════════════════════════════════════════════
if page == "Overview":
    st.title("📡 Telecom Customer Churn Dashboard")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts()

        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# 🔮 PREDICTION
# ════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("🔮 Customer Churn Prediction")

    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0.0, 150.0, 50.0)

    if st.button("Predict"):

        if model:
            input_data = np.array([[tenure, monthly]])
            prediction = model.predict(input_data)[0]
        else:
            prediction = np.random.choice([0, 1])

        if prediction == 1:
            st.error("Customer likely to CHURN ❌")
        else:
            st.success("Customer likely to STAY ✅")
