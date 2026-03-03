import streamlit as st
import requests
from auth_utils import get_api_url

st.set_page_config(page_title="Retail Intelligence OS", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

st.title("🚀 Retail Optimisation Engine")
st.subheader("A Production-Grade Personalization & Pricing Framework")

st.write("### 🟢 System Vitality")
try:
    # Use helper to get the health endpoint
    health_url = get_api_url("health")
    health = requests.get(health_url).json()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("API Status", "Online")
    c2.metric("Inference Engine", "ONNX Runtime")
    c3.metric("Vector DB", health.get("database", "Connected").capitalize())
except Exception:
    st.error("Backend connection required to view live status.")

st.divider()

st.markdown("""
### 🏗️ Architectural Excellence
- **Neural Retrieval:** Two-Tower PyTorch model exported to **ONNX**.
- **Deep Re-Ranking:** **XGBoost** classifier for profit/relevance balance.
- **Elastic Infrastructure:** **Redis** and **pgvector** integration.
""")

if st.button("Enter Secure Intelligence Portal", type="primary", use_container_width=True):
    st.switch_page("pages/Auth.py")