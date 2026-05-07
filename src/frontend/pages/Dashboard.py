import streamlit as st
import pandas as pd
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

# --- ADMIN SECTION ---
try:
    metrics_res = authenticated_request("GET", "metrics/system")
    
    if metrics_res.status_code == 200:
        st.title("🛡️ Admin Command Center")
        m = metrics_res.json()
        c1, c2 = st.columns(2)
        c1.metric("Redis Latency", f"{m['redis_latency_ms']}ms")
        c2.metric("System Health", m['status'].upper())
        tabs = st.tabs(["💰 Pricing Optimizer", "📈 Experiment Analytics"])

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
    st.caption("Demo flow: view -> click -> cart_add -> purchase. These events feed A/B metrics.")
    
    cols = st.columns(5)
    for i, item in enumerate(data['recommendations']):
        with cols[i]:
            with st.container(border=True):
                st.write(f"**{item['product_name']}**")
                st.write(f"Price: :green[${item['price']}]")
                st.write(f"Score: `{item['score']}`")
                if st.button("👀 View", key=f"v_{item['product_id']}"):
                    payload = {
                        "user_id": int(u_id), 
                        "product_id": int(item['product_id']), 
                        "event_type": "view", 
                        "experiment_group": display_group, 
                        "revenue": 0.0,
                        "margin": float(item['margin'])
                    }
                    authenticated_request("POST", "events", json=payload)
                    st.toast("View event captured.")

                if st.button("🖱️ Click", key=f"c_{item['product_id']}"):
                    payload = {
                        "user_id": int(u_id),
                        "product_id": int(item['product_id']),
                        "event_type": "click",
                        "experiment_group": display_group,
                        "revenue": 0.0,
                        "margin": float(item['margin'])
                    }
                    authenticated_request("POST", "events", json=payload)
                    st.toast("Click event captured.")

                if st.button("🧺 Add to Cart", key=f"a_{item['product_id']}"):
                    payload = {
                        "user_id": int(u_id),
                        "product_id": int(item['product_id']),
                        "event_type": "cart_add",
                        "experiment_group": display_group,
                        "revenue": 0.0,
                        "margin": float(item['margin'])
                    }
                    authenticated_request("POST", "events", json=payload)
                    st.toast("Cart event captured.")

                if st.button("🛒 Purchase", key=f"p_{item['product_id']}"):
                    payload = {
                        "user_id": int(u_id),
                        "product_id": int(item['product_id']),
                        "event_type": "purchase",
                        "experiment_group": display_group,
                        "revenue": float(item['price']),
                        "margin": float(item['margin'])
                    }
                    authenticated_request("POST", "events", json=payload)
                    st.toast(f"Purchase captured. ${item['price']} attributed to {display_group}.")

    if st.button("🔁 Refresh Recommendations from Latest Context"):
        rec_res = authenticated_request("POST", f"recommendations?user_id={u_id}&top_k=5")
        if rec_res.status_code == 200:
            st.session_state.current_recs = rec_res.json()
            st.success("Recommendations refreshed.")

if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("Landing.py")