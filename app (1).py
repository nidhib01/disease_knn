import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os

st.title("🩺 Cancer Clinical Biomarkers Classifier")
st.write("""
This interactive app uses a **K-Nearest Neighbors (KNN)** model to classify clinical states 
based on physical and biological biomarkers.
""")

# 1. Load Data & Train model
@st.cache_resource
def load_and_train_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"
    df = pd.read_csv(url)
    
    original_cols = ['age', 'bmi', 'glucose', 'insulin', 'homa', 'leptin', 'adiponectin', 'resistin', 'mcp_1']
    X = df[original_cols]
    y = df['classification']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)
    return knn, scaler

model, scaler = load_and_train_model()

# 2. Sidebar Input Form
st.sidebar.header("Input Patient Biomarkers")
age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=50)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=10, max_value=300, value=90)
insulin = st.sidebar.number_input("Insulin (µU/mL)", min_value=1.0, max_value=100.0, value=10.0)
homa = st.sidebar.number_input("HOMA Index", min_value=0.1, max_value=30.0, value=2.0)
leptin = st.sidebar.number_input("Leptin (ng/mL)", min_value=1.0, max_value=120.0, value=20.0)
adiponectin = st.sidebar.number_input("Adiponectin (µg/mL)", min_value=1.0, max_value=50.0, value=10.0)
resistin = st.sidebar.number_input("Resistin (ng/mL)", min_value=1.0, max_value=100.0, value=12.0)
mcp_1 = st.sidebar.number_input("MCP-1 (pg/mL)", min_value=10.0, max_value=2000.0, value=500.0)

# 3. Prediction Pipeline
if st.button("Predict Diagnostic Class"):
    input_features = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]])
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.success("🟢 Predicted: Class 1 (Healthy Controls)")
    else:
        st.warning("🔴 Predicted: Class 2 (Patients/Abnormal Clinical State)")
        
    st.write(f"**Confidence Score (Class 1 vs Class 2):** {probabilities[0]*100:.1f}% vs {probabilities[1]*100:.1f}%")
