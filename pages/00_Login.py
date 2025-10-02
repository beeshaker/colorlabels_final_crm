# pages/00_Login.py — Minimal (no CSS/HTML)
import time
import logging
import streamlit as st
from conn import Conn
from auth import Auth

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Login - Sales Dashboard", page_icon="🔐", layout="centered")

# ===================== SESSION INIT =====================
Auth.initialize_session()


# ===================== DB CONNECTION (CACHED) =====================
@st.cache_resource
def get_connection():
    try:
        return Conn()
    except Exception as e:
        logger.error(f"[DB] Connection init failed: {e}")
        return None

conn = get_connection()
if conn is None:
    st.error("⚠️ Unable to connect to database")
    st.info("Please check your database connection and try again")
    st.stop()

# ===================== ALREADY LOGGED IN =====================
if st.session_state.get("logged_in", False):
    st.success(f"✅ You are already logged in as **{st.session_state.username}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.switch_page("pages/01_Dashboard.py")
    with col2:
        if st.button("📘 Bookings", use_container_width=True):
            st.switch_page("pages/Bookings_Compare.py")
    with col3:
        if st.button("🎯 Targets", use_container_width=True):
            st.switch_page("pages/05_Targets.py")
    st.stop()

# ===================== LOGIN FORM =====================
st.title("🔐 Sales Dashboard Login")
st.write("Enter your credentials to access the dashboard.")

with st.form("login_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        username = st.text_input("Username", placeholder="Enter your username").strip()
    with c2:
        password = st.text_input("Password", type="password", placeholder="Enter your password")

    submitted = st.form_submit_button("🔓 Login", type="primary", use_container_width=True)

if submitted:
    if not username:
        st.error("⚠️ Please enter your username")
    elif not password:
        st.error("⚠️ Please enter your password")
    else:
        with st.spinner("🔄 Authenticating..."):
            time.sleep(0.4)
            try:
                ok, role = conn.authenticate_user(username, password)
                if ok:
                    st.session_state.username = username.upper()
                    st.session_state.role = role
                    st.session_state.logged_in = True
                    st.session_state.last_activity = time.time()

                    try:
                        conn.write_audit(st.session_state.username, "login", extra=f"Login ok, role={role}")
                    except Exception as e:
                        logger.warning(f"[AUDIT] write failed: {e}")

                    st.success(f"✅ Welcome, {st.session_state.username}!")
                    st.balloons()
                    st.switch_page("pages/01_Dashboard.py")
                else:
                    try:
                        conn.write_audit(username, "login_failed", extra="Invalid credentials")
                    except Exception as e:
                        logger.warning(f"[AUDIT] write failed: {e}")
                    st.error("❌ Invalid username or password")
            except Exception as e:
                logger.error(f"[LOGIN] error: {e}")
                st.error("⚠️ An error occurred during login. Please try again.")
                st.exception(e)

st.divider()
st.caption("Sales Dashboard v1.0.0 | © 2025")
