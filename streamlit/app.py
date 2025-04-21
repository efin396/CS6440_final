import subprocess
import streamlit as st
import threading
import os


def start_fastapi():
    subprocess.Popen(["python", os.path.join("src", "app.py")])  # Start FastAPI

start_fastapi()

# Simulate session state login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Sign-in page logic
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "doctor" and password == "123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials.")
else:
    st.switch_page("pages/search.py")
