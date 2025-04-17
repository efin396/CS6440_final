import streamlit as st
import requests
import pandas as pd
import math 
import json
FASTAPI_URL = "http://localhost:8000"
MODEL_ENDPOINT = "/inference_df"
import datetime

def preprocess_patient_for_model(patient, conditions):

    birthdate = pd.to_datetime(patient.get("BIRTHDATE"))
    deathdate = pd.to_datetime(datetime.date.today())
    age = (deathdate - birthdate).days // 365
     

    ethnicity = patient['ETHNICITY']
    gender = patient['GENDER']
    income = patient['INCOME']
    marital = patient['MARITAL']
    race = patient['RACE']

    opiate_dependent = conditions[conditions['CODE'].isin([55680006, 6525002])]  # This code signifies drug overdose risk
    chronic_pain = conditions[conditions['CODE'].isin([82423001])]  # Chronic Pain
    chronic_migraine = conditions[conditions['CODE'].isin([124171000119105])]
    imapcted_molars = conditions[conditions['CODE'].isin([196416002])]

    print("here")
    opiate_ids =  set(opiate_dependent[opiate_dependent["PATIENT"] == patient["Id"]])
    pain_ids = set(chronic_pain[chronic_pain["PATIENT"] == patient["Id"]])
    migraine_ids = set(chronic_migraine[chronic_migraine["PATIENT"] == patient["Id"]])
    molar_ids = set(imapcted_molars[imapcted_molars["PATIENT"] == patient["Id"]])
    print("pain_ids")
    print(molar_ids)
    dependent = len(opiate_ids) > 0
    chronic_pain = len(pain_ids) > 0
    chronic_migraine = len(migraine_ids) > 0
    imapcted_molars = len(molar_ids) > 0

    return [[
        marital,
        race, 
        ethnicity,
        gender,
        int(income),
        int(age),
        int(chronic_pain),
        int(chronic_migraine),
        int(imapcted_molars),
        int(dependent)
 
    ]]

def sanitize_json_compat(data):
    # Replace NaN, inf, -inf with None (or 0, if you'd rather impute)
    def clean(val):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None  # or 0
        return val
    return [[clean(v) for v in row] for row in data]

def patient_risk_assessment():
    st.title("Patient Risk Assessment")
    patient = st.session_state.get("selected_patient")
    
    if not patient:
        st.error("No patient selected. Please return to the search page.")
        if st.button("Back to Search"):
            st.switch_page("pages/search.py")
        return

    st.header(f"Patient: {patient['name']}")

    try:
        conditions_df = pd.read_csv('/home/nor/Documents/Spring2025/Health_Info/Project/CS6440Project/data/csv/combined_conditions.csv')
        input_data = preprocess_patient_for_model(patient, conditions_df)
        input_data = sanitize_json_compat(input_data)
        
        print("Payload:")
        print(json.dumps({"data": input_data}, indent=2)) 
        response = requests.post(
            f"{FASTAPI_URL}{MODEL_ENDPOINT}",
            json={"data": input_data}
        )
        response.raise_for_status()
        prediction = response.json()

        st.markdown("### Overall Risk Assessment")
        st.success(f"Predicted Risk Score: **{prediction}**")

    except Exception as e:
            st.error(f"Failed to get prediction from API: {e}")
            return

    # A button to return to the search page
    if st.button("Back to Search"):
        st.switch_page("pages/search.py")

patient_risk_assessment()