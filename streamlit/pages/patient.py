import streamlit as st
import requests
import pandas as pd
import math 
import json
FASTAPI_URL = "http://localhost:8000"
MODEL_ENDPOINT = "/inference_df"
import datetime

def preprocess_patient_for_model(patient, conditions):

    if not isinstance(['MARITAL'], str):
        patient['MARITAL'] = 'N'

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
        conditions_df = pd.read_csv('data/csv/combined_conditions.csv')
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
        print(prediction)
        score  = prediction['predictions'][0][0]
        st.markdown("### Overall Risk Assessment")
        if score < 0.2:
            prediction = "Very Low Risk"
            st.success(f"Predicted Risk Level: **{prediction}**")
            low_risk_provider_msg = (
                "Patient is assessed to have a low predicted risk for opioid-related complications.\n\n"
                "This suggests that, with appropriate prescribing practices and patient education, opioid therapy may be a viable option.\n\n"
                "Continue to follow standard monitoring protocols, assess for any emerging risk factors, and encourage open communication with the patient regarding medication use and potential side effects."
            )
            st.text(low_risk_provider_msg)
        elif score < 0.4:
            prediction  = "Low Risk"
            st.info(f"Predicted Risk Level: **{prediction}**")
            low_risk_provider_msg = (
                "Patient is assessed to have a low predicted risk for opioid-related complications.\n\n"
                "This suggests that, with appropriate prescribing practices and patient education, opioid therapy may be a viable option.\n\n"
                "Continue to follow standard monitoring protocols, assess for any emerging risk factors, and encourage open communication with the patient regarding medication use and potential side effects."
            )
            st.text(low_risk_provider_msg)
        elif score < 0.6:
            prediction =  "Moderate Risk"
            st.warning(f"Predicted Risk Level: **{prediction}**")
            low_risk_provider_msg = (
                "Patient is assessed to have a moderate predicted risk for opioid-related complications.\n\n"
                "This suggests that, with appropriate prescribing practices and patient education, opioid therapy may be a viable option but take precautions are necessary.\n\n"
                "Continue to follow standard monitoring protocols, assess for any emerging risk factors, and encourage open communication with the patient regarding medication use and potential side effects."
            )
            st.text(low_risk_provider_msg)
        elif score < 0.9:
            prediction =  "Moderate-High Risk"
            st.warning(f"Predicted Risk Level: **{prediction}**")
            moderate_risk_provider_msg = (
                "Patient is assessed to have a moderate predicted risk for opioid-related complications.\n\n"
                "Caution is advised when considering opioid therapy. Evaluate alternative pain management strategies where appropriate.\n\n"
                "If opioids are prescribed, implement enhanced monitoring protocols, set clear treatment goals, and ensure the patient is educated on safe use, storage, and disposal."
            )

            st.text(moderate_risk_provider_msg)
        else:
            prediction = "High Risk"
            st.error(f"Predicted Risk Level: **{prediction}**")

            high_risk_provider_msg = (
                "Patient is assessed to have a high predicted risk for opioid-related complications.\n\n"
                "Opioid prescribing should generally be avoided unless absolutely necessary. Prioritize non-opioid pain management options and consider consulting a pain specialist.\n\n"
                "If opioids must be used, strict monitoring, risk mitigation strategies, and frequent follow-up are essential. Document justification thoroughly and involve the patient in shared decision-making."
            )

            st.text(high_risk_provider_msg)

        
        

    except Exception as e:
            st.error(f"Failed to get prediction from API: {e}")
            return

    # A button to return to the search page
    if st.button("Back to Search"):
        st.switch_page("pages/search.py")

patient_risk_assessment()
