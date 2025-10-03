# menu.py
from pathlib import Path
import streamlit as st
try:
    # Newer Streamlit exposes this; if not present we'll just catch Exception
    from streamlit.errors import StreamlitPageNotFoundError
except Exception:  # pragma: no cover
    StreamlitPageNotFoundError = Exception  # fallback

# -------- helpers --------
def _exists(rel_path: str) -> bool:
    """Check if a page exists relative to the app working directory."""
    return (Path.cwd() / rel_path).exists()

def _first_existing(candidates: list[str]) -> str | None:
    for p in candidates:
        if _exists(p):
            return p
    return None

def _safe_page_link(candidates: list[str], label: str):
    """
    Link to the first existing candidate path; if none exist,
    skip without throwing StreamlitPageNotFoundError.
    """
    page = _first_existing(candidates)
    if not page:
        # Uncomment to show a hint instead of silently skipping:
        # st.sidebar.caption(f"⚠️ Missing page: {candidates[0]}")
        return
    try:
        st.sidebar.page_link(page, label=label)
    except StreamlitPageNotFoundError:
        # Registered pages don’t include this script (e.g., running from a different entrypoint);
        # skip gracefully.
        pass
    except Exception as e:
        # Any other unexpected error – show a tiny hint but keep the app running.
        st.sidebar.caption(f"⚠️ Couldn’t link {label}: {e}")

def menu():
    # Guard
    if not st.session_state.get("logged_in", False):
        st.warning("🔐 Please log in to continue.")
        st.stop()

    username = st.session_state.get("username", "USER")
    st.sidebar.markdown(f"👤 **User:** `{username}`")
    st.sidebar.divider()

    # --- Define your menu with candidate paths (to handle case/rename differences) ---
    _safe_page_link(
        ["pages/01_Dashboard.py", "pages/01_dashboard.py"],
        label="🏠 Dashboard"
    )
    _safe_page_link(
        ["pages/02_Bookings_compare.py", "pages/02_Bookings.py", "pages/Bookings_Compare.py"],
        label="📘 Bookings"
    )
    _safe_page_link(
        ["pages/03_Targets.py", "pages/03_targets.py"],
        label="🎯 Targets"
    )
    
    _safe_page_link(
        # Your repo showed different names across messages; try common variants:
        ["pages/05_Ai_analysis.py", "pages/05_AI_Analysis.py", "pages/AI_Analysis.py", "pages/AI_Insights.py"],
        label="🤖 AI Analysis"
    )

    st.sidebar.divider()
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.success("Logged out successfully!")
        # Adjust if your login file is different:
        _safe_page_link(["pages/00_login.py", "pages/00_Login.py"], label="🔑 Back to Login")
        # If you prefer an immediate redirect instead of a link:
        try:
            st.switch_page("pages/00_login.py")
        except Exception:
            pass
