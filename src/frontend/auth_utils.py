import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api/v1"

def login_user(username, password):
    """Sends credentials to FastAPI and stores JWT in Session State."""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/token",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            # We assume your backend returns {"access_token": "...", "role": "..."}
            # If your backend doesn't return 'role', you would decode it from the JWT here
            st.session_state.token = data["access_token"]
            st.session_state.role = data.get("role", "viewer")
            st.session_state.logged_in = True
            return True
        return False
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return False

def authenticated_get(endpoint):
    """Helper to make GET requests with the Bearer token."""
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers)
    return response.json() if response.status_code == 200 else None