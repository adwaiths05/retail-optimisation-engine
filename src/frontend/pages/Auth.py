import streamlit as st
import requests

st.set_page_config(page_title="Auth Portal", layout="centered", initial_sidebar_state="collapsed")
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

st.title("🔐 Identity Verification")

if "logged_in" not in st.session_state:
    st.session_state.update({"logged_in": False, "token": None, "role": None})

# Redirect if already logged in
if st.session_state.logged_in:
    st.switch_page("pages/Dashboard.py")

with st.container(border=True):
    st.write("Please enter your credentials to access the Intelligence OS.")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    
    if st.button("Authenticate & Enter", type="primary", use_container_width=True):
        try:
            # API CALL: /api/v1/auth/token
            res = requests.post("http://localhost:8000/api/v1/auth/token", 
                                json={"username": user, "password": pwd})
            
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data["access_token"]
                # Role assignment based on backend demo logic
                st.session_state.role = "admin" if user == "admin" else "viewer"
                st.session_state.logged_in = True
                
                st.success("Identity Verified! Redirecting...")
                st.balloons()
                st.rerun() # Immediate transition
            else:
                st.error("Invalid Credentials. Please try again.")
        except Exception:
            st.error("Backend Connection Failed. Ensure the FastAPI server is running.")

