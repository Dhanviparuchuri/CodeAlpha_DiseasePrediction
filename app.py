import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("diabetes_prediction_model.pkl")

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction")
st.write(
    "Enter the measurements below to see the prediction from our "
    "student machine-learning model."
)

st.warning("This is not a medical diagnosis or a substitute for advice from a healthcare professional.")

st.header("Enter details")

with st.form("prediction_form"):
    pregnancies = st.number_input(
        "Number of pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose level",
        min_value=1.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood pressure",
        min_value=1.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin thickness",
        min_value=1.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    insulin = st.number_input(
        "Insulin",
        min_value=1.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Keep the same feature names and order used when training
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction result")

    if prediction == 1:
        st.error("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No diabetes")

    st.write(f"Model-estimated probability for class 1: {probability:.1%}")
    st.caption(
        "This probability is a model output, not a person's actual "
        "medical risk. Please consult a qualified healthcare professional "
        "for medical interpretation."
    )