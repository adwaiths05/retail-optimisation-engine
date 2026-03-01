import streamlit as st

st.set_page_config(page_title="Retail AI", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS to hide the sidebar completely on this page
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

st.title("🌐 Retail Intelligence OS")
st.subheader("Next-Generation Demand & Pricing Engine")

st.markdown("""
This platform integrates **Real-time Vector Search** and **XGBoost Reranking** to automate retail decision-making at scale.

* **Neural Retrieval:** ONNX-powered user-intent mapping.
* **Elastic Pricing:** Dynamic margin adjustment algorithms.
* **Streamlined Security:** Industrial-grade JWT & RBAC.
""")

st.info("To explore the engine's capabilities, please authenticate via the secure portal.")

if st.button("Enter Secure Portal", type="primary", use_container_width=True):
    st.switch_page("pages/Auth.py")