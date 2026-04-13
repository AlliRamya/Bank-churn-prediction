# Bank Customer Churn Prediction

## 📌 Overview
This project predicts whether a customer will leave the bank using machine learning models.

## 📊 Dataset
- 10,000 customer records
- 18 features including credit score, balance, geography, etc.

## ⚙️ Models Used
- Random Forest
- XGBoost

## 🚀 Results
- Accuracy: 83%
- ROC-AUC: 0.77
- Recall (Churn class): 68%

## 🛠️ Features
- Handled class imbalance using SMOTE
- Feature engineering (BalanceSalaryRatio, TenureAgeRatio)
- Model evaluation using F1-score, ROC-AUC

## 💻 Tech Stack
Python, Scikit-learn, XGBoost, Pandas, NumPy, Streamlit

## ▶️ How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run:
   streamlit run app.py
