import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# Streamlit Page Settings
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Fill the details below to predict whether the passenger would have survived.")

# User Inputs
name = st.text_input("Passenger Name", "")

sex = st.selectbox("Gender", ["Male", "Female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibSp = st.number_input("Number of Siblings Onboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Onboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C = Cherbourg", "Q = Queenstown", "S = Southampton"])

# Convert Inputs
embarked = embarked[0]  # Get first character (C/Q/S)
sex = 1 if sex == "Female" else 0
fare_log = np.log1p(fare)
familysize = sibSp + parch + 1
isalone = 1 if familysize == 1 else 0

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

pclass_2 = 1 if pclass == 2 else 0
pclass_3 = 1 if pclass == 3 else 0

# Prepare input for model
inputData = pd.DataFrame([[
    sex, age, embarked_Q, embarked_S, pclass_2, pclass_3, familysize, isalone, fare_log
]], columns=['sex','age','embarked_Q','embarked_S','pclass_2','pclass_3','familysize','isalone','fare_log'])

# Predict Button
if st.button("Predict Survival"):
    prediction = model.predict(inputData)[0]
    probability = model.predict_proba(inputData)[0][1] * 100

    st.subheader(f"Prediction for **{name}**")
    
    if prediction == 1:
        st.success(f"✅ *Survived* — Probability: **{probability:.2f}%**")
    else:
        st.error(f"❌ *Did Not Survive* — Probability: **{probability:.2f}%**")

    st.write("---")
    st.caption("Model: Random Forest Classifier | Data: Titanic Dataset")
