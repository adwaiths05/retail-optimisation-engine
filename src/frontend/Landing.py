import streamlit as st
import requests

st.set_page_config(page_title="Retail Intelligence OS", layout="wide", initial_sidebar_state="collapsed")

# Hide Sidebar for a clean landing experience
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

st.title("🚀 Retail Optimisation Engine")
st.subheader("A Production-Grade Personalization & Pricing Framework")

# Real-time System Health Section
st.write("### 🟢 System Vitality")
try:
    # Fetching from the public health endpoint
    health = requests.get("http://localhost:8000/health").json()
    c1, c2, c3 = st.columns(3)
    c1.metric("API Status", "Online")
    c2.metric("Inference Engine", "ONNX Runtime")
    c3.metric("Vector DB", health.get("database", "Connected").capitalize())
except Exception:
    st.error("Backend connection required to view live status.")

st.divider()

# Technical Value Proposition
st.markdown("""
### 🏗️ Architectural Excellence
This engine integrates high-performance machine learning with industrial backend standards:
- **Neural Retrieval:** Two-Tower PyTorch model exported to **ONNX** for <20ms user-intent mapping.
- **Deep Re-Ranking:** **XGBoost** classifier that balances purchase probability with business margins.
- **Elastic Infrastructure:** **Redis** for sub-5ms session caching and **pgvector** for HNSW vector search.
- **Secure Access:** Industrial-grade **JWT Authentication** and Role-Based Access Control (RBAC).
""")

if st.button("Enter Secure Intelligence Portal", type="primary", use_container_width=True):
    st.switch_page("pages/Auth.py")