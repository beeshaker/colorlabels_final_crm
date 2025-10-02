# menu.py
import streamlit as st

def menu():
    # Guard
    if not st.session_state.get("logged_in", False):
        st.warning("🔐 Please log in to continue.")
        st.stop()

    username = st.session_state.get("username", "USER")
    st.sidebar.markdown(f"👤 **User:** `{username}`")
    st.sidebar.divider()

    # ——— Only these four pages ———
    st.sidebar.page_link("pages/01_Dashboard.py", label="🏠 Dashboard")
    st.sidebar.page_link("pages/02_Bookings_Compare.py", label="📘 Bookings")
    st.sidebar.page_link("pages/03_Targets.py", label="🎯 Targets")

    # Adjust the filename below to match your AI page exactly
    st.sidebar.page_link("pages/04_Ai_analysis.py", label="🤖 AI Analysis")
    # If your file is named differently, e.g. "AI_Analysis.py":
    # st.sidebar.page_link("pages/AI_Analysis.py", label="🤖 AI Analysis")

    st.sidebar.divider()
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.success("Logged out successfully!")
        st.switch_page("pages/00_login.py")  # or wherever your login lives
