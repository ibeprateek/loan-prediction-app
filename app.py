import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    'Age': [25, 35, 45, 20, 30, 50, 23, 40],
    'Income': [30000, 60000, 80000, 20000, 50000, 90000, 35000, 70000],
    'CreditScore': [650, 700, 750, 600, 720, 800, 680, 710],
    'LoanAmount': [100000, 200000, 250000, 80000, 150000, 300000, 120000, 220000],
    'LoanTerm': [2, 5, 10, 1, 3, 7, 2, 6],
    'Approved': [0, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# -----------------------------
# Train Model
# -----------------------------
X = df[['Age', 'Income', 'CreditScore', 'LoanAmount', 'LoanTerm']]
y = df['Approved']

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# UI Design
# -----------------------------
st.title("🏦 Loan Approval Prediction System")

st.write("Enter customer details:")

age = st.slider("Age", 18, 60, 30)
income = st.number_input("Income", value=50000)
credit = st.slider("Credit Score", 300, 850, 700)
loan_amt = st.number_input("Loan Amount", value=200000)
loan_term = st.slider("Loan Term (years)", 1, 10, 5)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):
    input_data = [[age, income, credit, loan_amt, loan_term]]
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Approval Probability: {prob[0][1]*100:.2f}%")