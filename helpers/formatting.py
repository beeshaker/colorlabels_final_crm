# helpers/formatting.py
def format_currency(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return "KSH 0"
    if v >= 1_000_000:
        return f"KSH {v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"KSH {v/1_000:.0f}K"
    return f"KSH {v:.0f}"

def conversion_emoji(pct: float) -> str:
    if pct >= 80:
        return "🟢"
    if pct >= 60:
        return "🟡"
    return "🔴"
