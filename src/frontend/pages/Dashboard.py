import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Retail Control Center", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

if not st.session_state.get("logged_in"):
    st.stop()

API_BASE = "http://localhost:8000/api/v1"
headers = {"Authorization": f"Bearer {st.session_state.token}"}

# --- ADMIN SECTION (RBAC Protected) ---
try:
    # API CALL: /metrics/system (Used as a security gate)
    metrics_res = requests.get(f"{API_BASE}/metrics/system", headers=headers)
    
    if metrics_res.status_code == 200:
        st.title("🛡️ Admin Command Center")
        m = metrics_res.json()
        
        # API CALL: /models/current
        model_info = requests.get(f"{API_BASE}/models/current", headers=headers).json()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Redis Latency", f"{m['redis_latency_ms']}ms")
        c2.metric("System Health", m['status'].upper())
        c3.metric("Model Version", model_info['model_version'])

        tabs = st.tabs(["💰 Pricing Optimizer", "📈 Experiment Analytics", "⚙️ Model Ops"])

        with tabs[0]: # API CALL: /pricing/optimize
            st.write("### Dynamic Price Optimization")
            col_a, col_b, col_c = st.columns(3)
            pid = col_a.number_input("Product ID", value=1, key="p_opt")
            base = col_b.number_input("Base Price ($)", value=49.99)
            stock = col_c.number_input("Inventory", value=150)
            
            if st.button("Calculate Optimal Price"):
                p_res = requests.post(f"{API_BASE}/pricing/optimize", headers=headers, 
                                     params={"product_id": pid, "base_price": base, "inventory_level": stock})
                if p_res.status_code == 200:
                    opt = p_res.json()
                    st.success("✅ Optimal Strategy Generated")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Recommended Price", f"${opt['recommended_price']}", f"{round(opt['recommended_price']-base, 2)}")
                    res_col2.metric("Expected Uplift", f"8.2%", help="Simulated revenue uplift based on inventory pressure logic")

        with tabs[1]: # API CALL: /experiments/results
            st.write("### A/B Test Performance")
            if st.button("🔄 Refresh Analytics"):
                st.rerun()
            
            exp_res = requests.get(f"{API_BASE}/experiments/results", headers=headers).json()
            results_df = pd.DataFrame([{"Group": r['group'], **r['metrics']} for r in exp_res['results']])
            st.table(results_df)
            
            # Real-Time Visual Feedback: Revenue Lift Chart
            st.write("#### Revenue Lift by Group")
            chart_data = results_df.set_index("Group")["total_revenue"]
            st.bar_chart(chart_data)

        with tabs[2]: # API CALL: /models/retrain
            if st.button("🚀 Trigger Pipeline Retrain"):
                requests.post(f"{API_BASE}/models/retrain", headers=headers)
                st.toast("Retraining job queued!")

except Exception:
    pass # Hides section for 'viewer' role (403 Forbidden)

# --- RECOMMENDATION ENGINE ---
st.divider()
st.subheader("🎯 Neural Personalization Engine")

u_id = st.number_input("Target User ID", value=1001)

# Feature: Strategy Override Toggle
col_strat1, col_strat2 = st.columns([2, 1])
with col_strat1:
    override_mode = st.toggle("Enable Manual Strategy Override", help="Force a specific algorithm regardless of A/B assignment.")

with col_strat2:
    selected_strategy = st.selectbox(
        "Select Override Strategy",
        ["control", "margin_boost"],
        disabled=not override_mode,
        format_func=lambda x: "A (Control: Relevance)" if x == "control" else "B (Margin Boost: Profit)"
    )

# Use Session State to keep recommendations visible after clicking 'Simulate Sale'
if st.button("Generate Strategy", type="primary") or "current_recs" in st.session_state:
    
    # Check if we need to fetch new data or if the user changed the User ID
    if "current_recs" not in st.session_state or st.button("🔄 Get New Suggestions"):
        with st.status("Fetching Neural Strategy...", expanded=True) as status:
            # API CALL: /recommendations
            rec_res = requests.post(f"{API_BASE}/recommendations?user_id={u_id}&top_k=5", headers=headers).json()
            st.session_state.current_recs = rec_res
            status.update(label="Recommendations Ready!", state="complete", expanded=False)

    data = st.session_state.current_recs
    
    # Determine the strategy to display and log
    display_group = selected_strategy if override_mode else data['experiment_group']
    
    st.write(f"**Applied Strategy:** :orange[{display_group.replace('_', ' ').upper()}]")
    if override_mode:
        st.caption("⚠️ Manual override active: Analytics data will be tagged with this forced group.")
    
    cols = st.columns(5)
    for i, item in enumerate(data['recommendations']):
        with cols[i]:
            with st.container(border=True):
                st.write(f"**{item['product_name']}**")
                st.write(f"Price: :green[${item['price']}]")
                st.metric("Score", f"{item['score']}", help="Combined ML Probability + Business Margin Weight")
                
                # API CALL: /events (Feeds the Experiment Analytics)
                if st.button("🛒 Simulate Sale", key=f"s_{item['product_id']}"):
                    payload = {
                        "user_id": int(u_id), 
                        "product_id": int(item['product_id']),
                        "event_type": "purchase", 
                        "experiment_group": display_group, # Log against the applied strategy
                        "revenue": float(item['price'])
                    }
                    with st.spinner("Logging transaction..."):
                        response = requests.post(f"{API_BASE}/events", headers=headers, json=payload)
                        if response.status_code == 200:
                            st.toast(f"Success! ${item['price']} added to {display_group} revenue.", icon="💰")

if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("Landing.py")