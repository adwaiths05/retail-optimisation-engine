import streamlit as st
import pandas as pd
import os
from auth_utils import authenticated_request

# Page configuration
st.set_page_config(
    page_title="Retail Control Center", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Hide Sidebar
st.markdown("<style> [data-testid='stSidebar'] {display: none} </style>", unsafe_allow_html=True)

# Authentication Guard
if not st.session_state.get("logged_in"):
    st.stop()

# --- MLOPS MONITORING UTILITY ---
def generate_drift_report(ref_list, curr_list):
    """
    Heavy lifting moved to Frontend to keep Backend < 500MB.
    Imports are scoped here so they only load when the button is clicked.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    
    ref_df = pd.DataFrame(ref_list)
    curr_df = pd.DataFrame(curr_list)
    
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)
    
    report_dir = "./mlops/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "drift_report.html")
    report.save_html(report_path)
    return report_path

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
            st.write("### 📜 Model Registry & Versioning")
            col_info, col_perf = st.columns([2, 1])
            
            with col_info:
                st.markdown("#### Deployment Details")
                st.markdown(f"""
                - **Model ID:** `{model_info.get('model_id', 'N/A')}`
                - **Version:** `{model_info.get('model_version', 'N/A')}`
                - **Framework:** `ONNX Runtime (XGBoost Pipeline)`
                - **Last Deployed:** `{model_info.get('deployed_at', 'Just Now')}`
                """)
                
                st.markdown("#### Input Features")
                features = model_info.get('features', ['price', 'category_id', 'avg_margin'])
                st.write(", ".join([f"`{f}`" for f in features]))

            with col_perf:
                st.markdown("#### Validation Metrics")
                for metric, value in perf.items():
                    st.metric(label=metric.upper(), value=f"{value:.4f}")

            st.divider()
            if st.button("🚀 Trigger Pipeline Re-run", use_container_width=True):
                with st.spinner("Initiating retraining job in MLflow..."):
                    authenticated_request("POST", "models/retrain")
                    st.success("Retraining job queued. Monitoring report will update shortly.")

        with tabs[3]:
            st.write("### 🔬 ML Observability & Drift")
            st.info("Analysis processed in Frontend container to keep Backend performance optimal.")

            if st.button("🔄 Analyze Live Production Drift"):
                with st.spinner("Fetching data from Neon and running statistical tests..."):
                    res = authenticated_request("GET", "models/monitoring-data")
                    if res.status_code == 200:
                        data = res.json()
                        if data['reference'] and data['current']:
                            generate_drift_report(data['reference'], data['current'])
                            st.success("✅ Analysis Complete!")
                        else:
                            st.warning("Insufficient data in database for analysis.")
                    else:
                        st.error("Could not fetch data from Backend.")

            # Load and display the report
            report_path = "./mlops/reports/drift_report.html"
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=1000, scrolling=True)
            else:
                st.caption("No report found. Click the button above to generate one.")

            st.write("---")
            st.write("**Prometheus Status:** Scraping active at `/metrics`")

except Exception as e:
    st.error(f"Error connecting to Admin Services: {e}")

# --- RECOMMENDATION ENGINE ---
st.divider()
st.subheader("🎯 Neural Personalization Engine")
u_id = st.number_input("Target User ID", value=1001)

# Persist recommendations in session state
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