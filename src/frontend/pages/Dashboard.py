import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Retail Control Center", layout="wide", initial_sidebar_state="expanded")

# --- AUTH CHECK ---
if not st.session_state.get("logged_in"):
    st.warning("⚠️ Unauthorized Access. Please authenticate on the Auth page.")
    st.stop()

# Configuration
API_BASE = "http://localhost:8000/api/v1"
headers = {"Authorization": f"Bearer {st.session_state.token}"}
role = st.session_state.role

# --- SIDEBAR: SYSTEM STATUS (Endpoint 1: /health) ---
with st.sidebar:
    st.title("👤 Session Info")
    st.write(f"**Role:** :blue[{role.upper()}]")
    
    try:
        health_res = requests.get("http://localhost:8000/health") # Endpoint 1
        if health_res.status_code == 200:
            st.success("🟢 API Status: Online")
        else:
            st.error("🔴 API Status: Offline")
    except:
        st.error("🔴 API Status: Connection Failed")

    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.switch_page("Landing.py")

# --- MAIN UI ---
st.title("📊 Retail Intelligence Dashboard")

# --- ADMIN SECTION (Endpoints: /metrics/system, /pricing/optimize, /models/retrain, /models/current) ---
if role == "admin":
    st.subheader("🛠️ Administrative Command Center")
    
    # 1. System Metrics (Endpoint 2: /metrics/system)
    try:
        m_res = requests.get(f"{API_BASE}/metrics/system", headers=headers)
        if m_res.status_code == 200:
            m = m_res.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Inference Latency", f"{m.get('latency', 0)}ms")
            c2.metric("Active Sessions", m.get("active_users", 0))
            c3.metric("CPU Usage", f"{m.get('cpu_load', 0)}%")
            c4.metric("Model State", "Healthy" if m.get('status') == 'healthy' else "Check Logs")
    except: st.warning("Metrics endpoint unreachable")

    tabs = st.tabs(["💰 Pricing Optimizer", "🧠 Model Lifecycle", "📈 A/B Insights"])

    with tabs[0]: # Endpoint 3: /pricing/optimize
        st.write("### Dynamic Price Optimization")
        col_a, col_b = st.columns(2)
        with col_a:
            b_price = st.number_input("Base Product Price ($)", value=49.99, step=1.0)
            inv = st.number_input("Current Inventory Level", value=150, step=10)
        with col_b:
            comp_p = st.number_input("Competitor Price ($)", value=45.00, step=1.0)
            margin = st.slider("Min Target Margin (%)", 5, 40, 15)

        if st.button("Calculate Optimized Price", type="primary"):
            payload = {"base_price": b_price, "inventory_level": inv, "competitor_price": comp_p, "target_margin": margin/100}
            res = requests.post(f"{API_BASE}/pricing/optimize", headers=headers, json=payload)
            if res.status_code == 200:
                opt = res.json()
                st.info(f"💡 Recommended Strategy: **{opt.get('strategy', 'Standard')}**")
                st.metric("New Optimized Price", f"${opt.get('optimized_price'):.2f}", 
                          f"{opt.get('optimized_price') - b_price:.2f}")
            else: st.error("Pricing API Error")

    with tabs[1]: # Endpoint 4 & 5: /models/current & /models/retrain
        st.write("### Model Registry & Maintenance")
        curr_res = requests.get(f"{API_BASE}/models/current", headers=headers)
        if curr_res.status_code == 200:
            info = curr_res.json()
            st.code(f"Current Model: {info.get('version')} | Deployed: {info.get('last_updated')}")
        
        if st.button("🚀 Trigger Model Retraining"):
            retrain_res = requests.post(f"{API_BASE}/models/retrain", headers=headers)
            if retrain_res.status_code == 200:
                st.toast("Retraining Pipeline Started!", icon="🔄")
            else: st.error("Retrain failed")

    with tabs[2]: # Endpoint 6: /experiments/results
        st.write("### A/B Testing Performance")
        exp_res = requests.get(f"{API_BASE}/experiments/results", headers=headers)
        if exp_res.status_code == 200:
            data = exp_res.json().get("results", [])
            if data:
                df = pd.DataFrame(data)
                fig = px.bar(df, x="model_name", y="ctr", color="model_name", title="CTR by Model Variant")
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No active experiments found.")

# --- COMMON SECTION (Endpoints: /recommendations, /events) ---
st.divider()
st.subheader("🎯 Neural Recommendation Engine")

# 1. Input for Recommendations (Endpoint 7: /recommendations)
col_req1, col_req2 = st.columns([1, 3])
with col_req1:
    target_user = st.number_input("Target User ID", value=1001, step=1)
    k_items = st.select_slider("Recommendation Count", options=[5, 10, 15, 20])

if st.button("Generate Recommendations", type="primary"):
    rec_res = requests.post(f"{API_BASE}/recommendations", headers=headers, 
                            json={"user_id": target_user, "top_k": k_items})
    
    if rec_res.status_code == 200:
        items = rec_res.json().get("recommendations", [])
        st.write(f"Showing results for User **{target_user}**")
        
        # Proper Grid Output (No JSON)
        grid = st.columns(5)
        for i, item in enumerate(items):
            with grid[i % 5]:
                with st.container(border=True):
                    st.write(f"**Product #{item['id']}**")
                    st.write(f"Price: :green[${item['price']}]")
                    st.caption(f"Match: {item['score']*100:.1f}%")
                    
                    # 2. Event Tracking (Endpoint 8: /events)
                    if st.button("Buy Now", key=f"buy_{item['id']}"):
                        ev_res = requests.post(f"{API_BASE}/events", headers=headers,
                                               json={"user_id": target_user, "item_id": item['id'], "event": "purchase"})
                        if ev_res.status_code == 200:
                            st.toast(f"Purchase logged for User {target_user}!", icon="💰")
    else:
        st.error("Could not fetch recommendations.")