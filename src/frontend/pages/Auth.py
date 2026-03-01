import streamlit as st
import requests

st.set_page_config(page_title="Auth Portal", layout="centered", initial_sidebar_state="collapsed")
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

st.title("🔐 Authentication")

if "logged_in" not in st.session_state:
    st.session_state.update({"logged_in": False, "token": None, "role": None})

with st.container(border=True):
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    
    if st.button("Authenticate", use_container_width=True):
        try:
            res = requests.post("http://localhost:8000/api/v1/auth/token", 
                                json={"username": user, "password": pwd})
            if res.status_code == 200:
                st.session_state.token = res.json()["access_token"]
                st.session_state.role = "admin" if user == "admin" else "viewer"
                st.session_state.logged_in = True
                st.success("Identity Verified!")
                st.balloons()
            else:
                st.error("Invalid Credentials")
        except:
            st.error("Backend Connection Failed")

# The button only appears AFTER successful login
if st.session_state.logged_in:
    if st.button("Go to Intelligence Dashboard 🚀", use_container_width=True):
        st.switch_page("pages/Dashboard.py")