import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load saved files
# ===============================
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Bank Churn Prediction", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")

# ===============================
# USER INPUTS
# ===============================
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Extra fields (IMPORTANT)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
complain = st.selectbox("Customer Complaint", [0, 1])
satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM"])
points = st.number_input("Points Earned", 0, 2000, 500)

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0

# Fill basic features
input_df["CreditScore"] = credit_score
input_df["Age"] = age
input_df["Tenure"] = tenure
input_df["Balance"] = balance
input_df["NumOfProducts"] = num_products
input_df["HasCrCard"] = has_card
input_df["IsActiveMember"] = is_active
input_df["EstimatedSalary"] = salary

# Geography Encoding
if "Geography_Germany" in input_df.columns:
    input_df["Geography_Germany"] = 1 if geography == "Germany" else 0
if "Geography_Spain" in input_df.columns:
    input_df["Geography_Spain"] = 1 if geography == "Spain" else 0

# Gender Encoding
if "Gender_Male" in input_df.columns:
    input_df["Gender_Male"] = 1 if gender == "Male" else 0

# Extra Features
if "Complain" in input_df.columns:
    input_df["Complain"] = complain

if "Satisfaction Score" in input_df.columns:
    input_df["Satisfaction Score"] = satisfaction

if "Point Earned" in input_df.columns:
    input_df["Point Earned"] = points

# Card Type Encoding
for col in input_df.columns:
    if "Card Type_" in col:
        input_df[col] = 0

if f"Card Type_{card_type}" in input_df.columns:
    input_df[f"Card Type_{card_type}"] = 1

# ===============================
# FINAL CHECK
# ===============================
input_df = input_df.fillna(0)

st.write("✅ Feature Count Expected:", len(feature_names))
st.write("✅ Feature Count Given:", input_df.shape[1])

# ===============================
# SCALING
# ===============================
features = scaler.transform(input_df)

# ===============================
# MODEL SELECTION
# ===============================
model_choice = st.radio("Choose Model", ["Random Forest", "XGBoost"])

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):

    if model_choice == "Random Forest":
        pred = rf_model.predict(features)[0]
        prob = rf_model.predict_proba(features)[0][1]
    else:
        pred = xgb_model.predict(features)[0]
        prob = xgb_model.predict_proba(features)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer will CHURN\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Customer will NOT churn\nProbability: {prob:.2f}")