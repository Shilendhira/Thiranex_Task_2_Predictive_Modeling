import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# Create the same training data used in the notebook
n = 500
age = np.random.randint(18, 61, n)
salary = np.random.randint(20000, 150001, n)
score = 0.06 * age + 0.000018 * salary + np.random.normal(0, 0.7, n)
purchase = (score > 3.2).astype(int)

X = pd.DataFrame({"Age": age, "EstimatedSalary": salary})
y = pd.Series(purchase)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.set_page_config(page_title="Customer Purchase Predictor", page_icon="🤖")

st.title("🤖 Customer Purchase Prediction")
st.write("Predict whether a customer is likely to purchase a product using a Random Forest model.")

age_input = st.number_input("Age", min_value=18, max_value=100, value=40)
salary_input = st.number_input("Estimated Salary", min_value=10000, max_value=500000, value=80000, step=1000)

if st.button("Predict"):
    customer = pd.DataFrame({
        "Age": [age_input],
        "EstimatedSalary": [salary_input]
    })

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase — probability: {probability:.2%}")
    else:
        st.warning(f"Unlikely to purchase — probability: {probability:.2%}")
