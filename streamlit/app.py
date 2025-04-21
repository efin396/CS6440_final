import subprocess
import streamlit as st
import threading


def start_fastapi():
    subprocess.run(["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"])

# Start FastAPI in a separate thread
fastapi_thread = threading.Thread(target=start_fastapi)
fastapi_thread.daemon = True  # This makes the thread terminate when the main program ends
fastapi_thread.start()
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
