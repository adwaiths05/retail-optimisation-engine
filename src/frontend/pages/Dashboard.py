import streamlit as st
import pandas as pd
from auth_utils import authenticated_request

st.set_page_config(page_title="Retail Control Center", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

if not st.session_state.get("logged_in"):
    st.stop()

# --- ADMIN SECTION ---
try:
    metrics_res = authenticated_request("GET", "metrics/system")
    
    if metrics_res.status_code == 200:
        st.title("🛡️ Admin Command Center")
        m = metrics_res.json()
        
        # Pull real metadata from the MLOps Registry
        model_info = authenticated_request("GET", "models/current").json()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Redis Latency", f"{m['redis_latency_ms']}ms")
        c2.metric("System Health", m['status'].upper())
        c3.metric("Live Model", model_info.get('model_version', 'N/A'))
        # Display AUC tracked via MLflow
        perf = model_info.get('performance', {})
        c4.metric("Training AUC", f"{perf.get('auc', 0):.4f}")

        tabs = st.tabs(["💰 Pricing Optimizer", "📈 Experiment Analytics", "⚙️ Model Ops", "🔬 ML Observability"])

        with tabs[0]:
            st.write("### Dynamic Price Optimization")
            col_a, col_b, col_c = st.columns(3)
            pid = col_a.number_input("Product ID", value=1, key="p_opt")
            base = col_b.number_input("Base Price ($)", value=49.99)
            stock = col_c.number_input("Inventory", value=150)
            
            if st.button("Calculate Optimal Price"):
                p_res = authenticated_request("POST", "pricing/optimize", 
                                             params={"product_id": pid, "base_price": base, "inventory_level": stock})
                if p_res.status_code == 200:
                    opt = p_res.json()
                    st.success("✅ Optimal Strategy Generated")
                    st.metric("Recommended Price", f"${opt['recommended_price']}")

        with tabs[1]:
            st.write("### A/B Testing Performance")
            exp_res = authenticated_request("GET", "experiments/results").json()
            results_df = pd.DataFrame([{"Group": r['group'], **r['metrics']} for r in exp_res['results']])
            st.table(results_df)
            st.bar_chart(results_df.set_index("Group")["total_revenue"])

        with tabs[2]:
            st.write("### Model Registry Metadata")
            st.json(model_info)
            if st.button("Trigger Pipeline Re-run"):
                authenticated_request("POST", "models/retrain")
                st.info("Retraining job queued in MLflow.")

        with tabs[3]:
            st.write("### Data Drift & Model Quality")
            st.info("Infrastructure monitored via Prometheus. Model quality monitored via Evidently.")
            
            # Link to the generated Evidently report
            st.markdown("#### [🔗 Open Latest Evidently Drift Report](./mlops/reports/drift_report.html)", unsafe_allow_html=True)
            
            st.write("---")
            st.write("**Prometheus Status:** Scraping active at `/metrics`")

except Exception as e:
    st.error(f"Error connecting to Admin Services: {e}")

# --- RECOMMENDATION ENGINE ---
st.divider()
st.subheader("🎯 Neural Personalization Engine")
u_id = st.number_input("Target User ID", value=1001)

if st.button("Generate Strategy", type="primary") or "current_recs" in st.session_state:
    if "current_recs" not in st.session_state or st.button("🔄 Get New Suggestions"):
        with st.status("Fetching Neural Strategy...") as status:
            rec_res = authenticated_request("POST", f"recommendations?user_id={u_id}&top_k=5").json()
            st.session_state.current_recs = rec_res
            status.update(label="Ready!", state="complete")

    data = st.session_state.current_recs
    display_group = data['experiment_group']
    
    st.write(f"User assigned to: **{display_group.replace('_', ' ').title()}**")
    
    cols = st.columns(5)
    for i, item in enumerate(data['recommendations']):
        with cols[i]:
            with st.container(border=True):
                st.write(f"**{item['product_name']}**")
                st.write(f"Price: :green[${item['price']}]")
                st.write(f"Score: `{item['score']}`")
                if st.button("🛒 Simulate Sale", key=f"s_{item['product_id']}"):
                    payload = {
                        "user_id": int(u_id), 
                        "product_id": int(item['product_id']), 
                        "event_type": "purchase", 
                        "experiment_group": display_group, 
                        "revenue": float(item['price'])
                    }
                    authenticated_request("POST", "events", json=payload)
                    st.toast(f"Event Captured! ${item['price']} attributed to {display_group}.")

if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("Landing.py")