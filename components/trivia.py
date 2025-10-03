# components/trivia.py
# Monthly AI-powered Kenya-focused label manufacturing SALES trivia (30/day pool)
# The user persona is a salesperson working for a label converter/manufacturer.
# Usage:
#   from components.trivia import show_monthly_trivia_widget
#   show_monthly_trivia_widget()

import os, json, random, datetime as dt
from zoneinfo import ZoneInfo
import streamlit as st

# --- Load environment variables from .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()  # will read .env file at project root
except ImportError:
    pass  # dotenv is optional if you only use st.secrets

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

TZ = ZoneInfo("Africa/Nairobi")
MODEL = "gpt-4o-mini"

# ---------- Fallback items (explicitly label-manufacturing sales) ----------
FALLBACK_POOL = [
    "💡 Pitch to beverage SMEs: switching from paper to PP labels cuts condensation smearing and reduces returns during cold-chain deliveries.",
    "❓ For yogurt cups in Nairobi MT, which boosts shelf pop more—spot UV on brand mark or matte + foil on the flavor callout?",
    "💡 Route sales tip: propose pre-printed QR reorder labels for top SKUs to reduce out-of-stocks in CBD and Westlands kiosks.",
    "❓ True/False: Bilingual (EN/SWA) usage icons on pesticide labels can reduce helpline calls and speed repeat orders.",
    "💡 For water brands in estates, tamper-evident neck labels + reorder QR often lift repeat purchases vs plain caps.",
    "❓ What cuts mis-scans more in general trade: wider barcode quiet zones or thicker bars on thermal stock?",
    "💡 Offer phased migration: paper → synthetic → specialty finishes for cosmetics; upsell via pilot SKUs before full rollout.",
    "❓ True/False: For chilled juice, waterproof PP with strong adhesive reduces label lift and keeps shelves tidy.",
    "💡 Industrial drums: chemical-resistant laminates reduce relabeling costs and strengthen procurement relationships.",
    "❓ For detergents targeting kiosks, which drives trial: bold color blocks or detailed copy on back label?",
]

# ---------- Helpers ----------
def _get_key():
    """
    Get OpenAI key safely.
    Priority: OS env/.env -> st.secrets (if present).
    Never raise if secrets.toml is missing.
    """
    # 1) .env / environment
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # 2) Streamlit secrets (won't crash if missing)
    try:
        return st.secrets["OPENAI_API_KEY"]  # may raise if secrets file absent
    except Exception:
        return None


def _month_key(d: dt.date) -> str:
    return d.strftime("%Y-%m")

def _fallback_30():
    return [random.choice(FALLBACK_POOL) for _ in range(30)]

def _ask_openai_for_30(month_key: str):
    """
    Ask OpenAI for 30 items tailored to a salesperson at a LABEL MANUFACTURER in Kenya.
    Must return a JSON array of 30 strings. Falls back to local items on failure.
    """
    api_key = _get_key()
    if (OpenAI is None) or (not api_key):
        return _fallback_30()

    client = OpenAI(api_key=api_key)

    system = (
        "You are a concise sales coach for **label manufacturing sales reps** in Kenya. "
        "Your audience sells self-adhesive labels and sleeves to FMCG, beverage, cosmetics, pharma, and industrial clients. "
        "Return EXACTLY 30 items as a JSON array of strings (no extra text). "
        "Each item must be ONE line (≤ 30 words), start with ❓ (question) or 💡 (tip). "
        "Focus on selling outcomes: prospecting, discovery, spec upgrades (paper→PP, PP→PE), adhesives, varnish/lamination, foil, spot UV, QR/USSD, barcode scanability, MOQ, lead times, proofs, compliance, GT vs MT realities, route selling, and upsell/cross-sell strategies. "
        "Use Kenya context naturally (Nairobi, Westlands, Industrial Area, Thika, Mombasa Road, general trade vs modern trade, kiosks/dukas, boda distribution, ambient vs chilled). "
        "Avoid brand names and exact percentages; keep claims generic but useful."
    )

    user = f"""
Month key: {month_key}

Output rules:
- JSON array of 30 strings ONLY. No markdown, no prose outside the array.
- Mix questions and tips; start each with ❓ or 💡.
- Make it obviously tied to selling **labels** (materials, finishes, adhesives, print effects, QR/USSD, barcode, compliance, line trials, pilot runs, MOQ/lead time).
- Emphasize conversations that a label rep would have with purchasing, production, brand, and QA teams.
- Keep it actionable in the Kenya field context (routes, store types, transport, humidity/condensation, chilled vs ambient).
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.9,
            max_tokens=1500,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        items = json.loads(text)
        items = [s.strip() for s in items if isinstance(s, str) and len(s.strip()) > 3]
        norm = []
        for s in items:
            if not (s.startswith("❓") or s.startswith("💡")):
                s = "💡 " + s
            norm.append(s if len(s) <= 240 else s[:237] + "...")
        if len(norm) >= 30:
            return norm[:30]
    except Exception:
        pass

    return _fallback_30()

def _ensure_month_pool():
    """Ensure a 30-item pool exists for the current month; refresh when month flips or pool is empty."""
    today = dt.datetime.now(TZ).date()
    mkey = _month_key(today)

    if "trivia_pool" not in st.session_state:
        st.session_state.trivia_pool = []
    if "trivia_month_key" not in st.session_state:
        st.session_state.trivia_month_key = ""
    if "trivia_last_shown_date" not in st.session_state:
        st.session_state.trivia_last_shown_date = None

    if (st.session_state.trivia_month_key != mkey) or (not st.session_state.trivia_pool):
        with st.spinner("Fetching monthly label-sales trivia…"):
            st.session_state.trivia_pool = _ask_openai_for_30(mkey)
        st.session_state.trivia_month_key = mkey
        st.session_state.trivia_last_shown_date = None

def _pop_today_item() -> str:
    """Pop exactly one item per local day. If revisited the same day, show the same item without popping."""
    today_str = dt.datetime.now(TZ).date().isoformat()
    if st.session_state.trivia_last_shown_date == today_str:
        return st.session_state.trivia_pool[0] if st.session_state.trivia_pool else random.choice(FALLBACK_POOL)

    if not st.session_state.trivia_pool:
        _ensure_month_pool()
    if not st.session_state.trivia_pool:
        return random.choice(FALLBACK_POOL)

    item = st.session_state.trivia_pool.pop(0)
    st.session_state.trivia_last_shown_date = today_str
    return item

# ---------- Public API ----------
def show_monthly_trivia_widget(title: str = "Sales Trivia"):
    """Render the daily trivia card and a refresh button to fetch a new batch of 30."""
    _ensure_month_pool()
    item = _pop_today_item()

    st.markdown(
        f"""
        <div style="
            margin-top:1rem;
            padding:1rem 1.25rem;
            border-radius:12px;
            background:#ffffff;
            box-shadow:0 8px 20px rgba(0,0,0,0.08);
            border:1px solid #eef2f7;">
            <div style="font-size:.9rem;font-weight:600;color:#6b7280;letter-spacing:.02em;">
                {title}
            </div>
            <div style="margin-top:.5rem;font-size:1.05rem;color:#111827;">
                {item}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

   
