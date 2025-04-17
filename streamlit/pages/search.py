import streamlit as st
import pandas as pd

def load_patients(csv_path="/home/nor/Documents/Spring2025/Health_Info/Project/CS6440Project/data/csv/combined_patients.csv"):
    df = pd.read_csv(csv_path)
    # Create full names from first and last name columns
    df['FIRST'] = df["FIRST"].str.replace('\d+', '', regex=True)
    df['LAST'] = df["LAST"].str.replace('\d+', '', regex=True)
    df["name"] = df["FIRST"]+ " " + df["LAST"]
    return df


def patient_search():
    st.title("Patient Search")

    patients_df = load_patients()
    PATIENTS = patients_df.to_dict(orient="records")

    query = st.text_input("Search for a patient by name")

    if query:
        results = [p for p in PATIENTS if query.lower() in p["name"].lower()]
    else:
        results = []

    # Display results only if there is a search query
    if query:
        if results:
            st.markdown("### Results")
            for patient in results:
                # Create a button for each patient
                if st.button(patient["name"], key=patient["Id"]):
                    st.session_state["selected_patient"] = patient
                    st.switch_page("pages/patient.py")
        else:
            st.info("No patients found. Try a different search query.")
    else:
        st.info("Enter a patient name to search.")

patient_search()
