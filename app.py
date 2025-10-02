# app.py — Home + Inline Login + Trivia (no CSS/HTML)
"""
Sales Dashboard Application
Homepage shows inline login (no redirect) + monthly trivia.
If authenticated, user sees the menu and can access dashboard pages.
"""

import time
import logging
import streamlit as st
from auth import Auth
from conn import Conn
from components import show_monthly_trivia_widget
from menu import menu  # <- call only after login

logger = logging.getLogger(__name__)

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

# ===================== AUTH'D USERS =====================
if st.session_state.get("logged_in", False):
    # ✅ Only call menu() when the user is logged in
    menu()

    # Show a simple dashboard landing on the homepage
    st.title("🏠 Dashboard")
    show_monthly_trivia_widget()
    st.divider()
    st.caption("© 2025 Sales Dashboard • Secure Access Required")
    st.stop()  # prevent the login form from rendering below

# ===================== HOMEPAGE (NOT LOGGED IN) =====================
st.title("📊 Sales Dashboard")
st.caption("Your comprehensive sales analytics platform")

left, right = st.columns([1, 1])

with left:
    st.subheader("🔐 Login")
    st.write("Enter your credentials to continue.")

    with st.form("home_login_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            username = st.text_input("Username", placeholder="Enter your username").strip()
        with c2:
            password = st.text_input("Password", type="password", placeholder="Enter your password")

        remember_me = st.checkbox("Remember me", value=False, help="(Coming soon)")
        submitted = st.form_submit_button("🔓 Login", type="primary", use_container_width=True)

    if submitted:
        if not username:
            st.error("⚠️ Please enter your username")
        elif not password:
            st.error("⚠️ Please enter your password")
        elif conn is None:
            st.error("❌ Unable to connect to database. Please try again later.")
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

                        # Best-effort audit
                        try:
                            conn.write_audit(st.session_state.username, "login", extra=f"Login ok, role={role}")
                        except Exception as e:
                            logger.warning(f"[AUDIT] write failed: {e}")

                        st.success(f"✅ Welcome, {st.session_state.username}!")
                        st.balloons()
                        # Send user straight to dashboard page
                        st.switch_page("pages/01_Dashboard.py")
                    else:
                        # Best-effort audit
                        try:
                            conn.write_audit(username, "login_failed", extra="Invalid credentials")
                        except Exception as e:
                            logger.warning(f"[AUDIT] write failed: {e}")

                        st.error("❌ Invalid username or password")
                        with st.expander("🆘 Having trouble logging in?"):
                            st.markdown(
                                """
- Username should be **UPPERCASE** (e.g., `JOHN`, not `john`)
- Password is **case-sensitive**
- Default password is usually lowercase username + `123`
- Contact your administrator if you've forgotten your password
                                """
                            )
                except Exception as e:
                    logger.error(f"[LOGIN] error: {e}")
                    st.error("⚠️ An error occurred during login. Please try again.")
                    st.exception(e)

with right:
    st.subheader("✨ Highlights")
    st.markdown(
        """
- 📈 Real-time sales tracking
- 📘 Bookings management
- 🎯 Customer targeting
- 📊 Performance analytics
        """
    )
    # Monthly Trivia Widget (kept on homepage)
    show_monthly_trivia_widget()

st.divider()
st.caption("© 2025 Sales Dashboard • Secure Access Required")
