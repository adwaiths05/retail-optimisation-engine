import streamlit as st
from auth_utils import login_user

st.set_page_config(page_title="Auth Portal", layout="centered", initial_sidebar_state="collapsed")
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

st.title("🔐 Identity Verification")

if "logged_in" not in st.session_state:
    st.session_state.update({"logged_in": False, "token": None, "role": None})

if st.session_state.logged_in:
    st.switch_page("pages/Dashboard.py")

with st.container(border=True):
    st.write("Please enter your credentials to access the Intelligence OS.")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    
    if st.button("Authenticate & Enter", type="primary", use_container_width=True):
        if login_user(user, pwd):
            st.success("Identity Verified! Redirecting...")
            st.balloons()
            st.rerun()
        else:
            st.error("Authentication failed. Check credentials or backend status.")