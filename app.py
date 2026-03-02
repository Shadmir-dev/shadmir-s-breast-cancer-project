import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("Breast Cancer Prediction App")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# User input
input_data = []
for feature in data.feature_names:
    value = st.number_input(feature, value=0.0)
    input_data.append(value)

if st.button("Predict"):
    prediction = model.predict([input_data])
    result = "Benign" if prediction[0] == 1 else "Malignant"
    st.success(f"Prediction: {result}")