import streamlit as st
import requests
import os

# --- CONFIGURATION ---
# Default to localhost for local development
# Use http://api:8000 for Docker Compose (service name)
# Use the Render URL for production deployment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

def get_api_url(endpoint: str):
    """Helper to construct endpoints."""
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]
    # Health check is usually at the root, others under /api/v1
    if endpoint == "health":
        return f"{BACKEND_URL}/health"
    return f"{API_BASE}/{endpoint}"

def login_user(username, password):
    """Sends credentials to FastAPI and stores JWT in Session State."""
    try:
        url = get_api_url("auth/token")
        response = requests.post(url, json={"username": username, "password": password})
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.role = "admin" if username == "admin" else "viewer"
            st.session_state.logged_in = True
            return True
        return False
    except Exception as e:
        st.error(f"Backend Connection Error: {e}")
        return False

def authenticated_request(method, endpoint, **kwargs):
    """Generic helper for authenticated requests."""
    headers = kwargs.get("headers", {})
    if "token" in st.session_state and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    kwargs["headers"] = headers
    url = get_api_url(endpoint)
    
    return requests.request(method, url, **kwargs)