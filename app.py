import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Title of the App
st.title("Titanic Survival Prediction - Machine Learning")
st.markdown("**Developed by Elaiyarani Soundararajan**")

# Load the trained model
model = joblib.load("models/best_xgboost_model.pkl")

# Sidebar for user input
st.sidebar.header("Passenger Information")

# User inputs
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])
familysize = st.sidebar.slider("Family Size", 1, 11, 1)
agegroup = st.sidebar.selectbox("Age Group", ["Child", "Teen", "Adult", "Senior"])
fareband = st.sidebar.selectbox("Fare Band", ["Low", "Mid", "High", "Very High"])

# Encoding user input
sex = 0 if sex == "Male" else 1
embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked] # type: ignore
title = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}[title] # type: ignore
agegroup = {"Child": 0, "Teen": 1, "Adult": 2, "Senior": 3}[agegroup] # type: ignore
fareband = {"Low": 0, "Mid": 1, "High": 2, "Very High": 3}[fareband] # type: ignore

# Create DataFrame for prediction
input_data = pd.DataFrame([[pclass, sex, embarked, title, familysize, agegroup, fareband]], 
                          columns=["Pclass", "Sex", "Embarked", "Title", "FamilySize", "AgeGroup", "FareBand"])

# Display Input Data
st.subheader("Passenger Information Summary")
st.table(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")

# Display Feature Importance
if st.button("Show Feature Importance"):
    fig, ax = plt.subplots(figsize=(6, 4))
    feature_importance = model.get_booster().get_score(importance_type="weight")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    labels, scores = zip(*sorted_features)
    ax.barh(labels, scores)
    ax.set_title("Feature Importance")
    st.pyplot(fig)