"""
Bot Dashboard — professional white theme.
Channels: WhatsApp, Messenger, Threads, Email.
"""
import sys, os, time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(__file__))

import store
from channels.base import Message
from channels.registry import get_channel
from crew.pipeline import Pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoBot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# White base, Inter typography, indigo accent, channel colors as identity only
CHANNEL_META = {
    "whatsapp":  {"label": "WhatsApp",  "color": "#25D366", "icon": "💬"},
    "email":     {"label": "Email",     "color": "#0072C6", "icon": "✉️"},
    "messenger": {"label": "Messenger", "color": "#0084FF", "icon": "🗨️"},
    "threads":   {"label": "Threads",   "color": "#000000", "icon": "🧵"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 16px;
    color: #111827 !important;
    background-color: #F6F8FA !important;
}
body {
    background: #F6F8FA !important;
}
div[data-testid="stAppViewContainer"],
div[data-testid="stMainContent"],
div[data-testid="stAppViewContainer"] > div,
div[data-testid="stMainContent"] > div,
.main .block-container,
main,
div[role="main"],
section[data-testid="stSidebar"] {
    background-color: #F6F8FA !important;
    color: #111827 !important;
}
.main .block-container {
    padding: 2.25rem 2.5rem 2rem 2rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    box-shadow: 2px 0 24px rgba(15, 23, 42, 0.08);
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.75rem 1.25rem !important;
}
section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 1.5rem;
}
.topbar-title {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.3px;
}
.topbar-sub {
    font-size: 14px;
    color: #6B7280;
    margin-top: 4px;
}

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.75rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
    min-width: 130px;
}
.stat-pill .num {
    font-size: 26px;
    font-weight: 700;
    color: #111827;
    line-height: 1.05;
}
.stat-pill .lbl {
    font-size: 12px;
    color: #6B7280;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Message card ── */
.msg-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 18px;
    border-left-width: 4px;
    transition: box-shadow 0.15s, transform 0.15s;
}
.msg-card:hover {
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
}

.msg-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.ch-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sender {
    font-weight: 700;
    font-size: 16px;
    color: #111827;
    flex: 1;
}
.ts {
    font-size: 12px;
    color: #6B7280;
    font-family: 'JetBrains Mono', monospace;
}

/* Status badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.badge-replied  { background: #DCFCE7; color: #166534; }
.badge-failed   { background: #FEE2E2; color: #991B1B; }
.badge-pending  { background: #FEF9C3; color: #854D0E; }

/* Message body */
.msg-body {
    font-size: 15px;
    color: #374151;
    line-height: 1.75;
    margin-bottom: 4px;
    white-space: pre-wrap;
}
.msg-subject {
    font-size: 12px;
    color: #6B7280;
    margin-bottom: 8px;
    font-style: italic;
}

/* Bot reply */
.bot-reply {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 14px;
    font-size: 14px;
    color: #475569;
    line-height: 1.75;
}
.bot-reply-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94A3B8;
    margin-bottom: 6px;
}

/* Section header */
.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #4B5563;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #E5E7EB;
}

/* Empty state */
.empty {
    text-align: center;
    padding: 48px 0;
    color: #9CA3AF;
    font-size: 14px;
}
.empty-icon { font-size: 32px; margin-bottom: 8px; }

/* Streamlit overrides */
[data-testid="stToolbar"],
[data-testid="stHeader"] {
    background: transparent !important;
}
div[data-testid="stTabs"] button {
    font-size: 14px;
    font-weight: 600;
}
div[data-testid="stExpander"],
div[data-testid="stExpander"] .stExpander {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}
.stButton button {
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

pipeline = Pipeline()


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        now = datetime.utcnow()
        diff = int((now - dt).total_seconds())
        if diff < 60:    return "just now"
        if diff < 3600:  return f"{diff//60}m ago"
        if diff < 86400: return f"{diff//3600}h ago"
        return dt.strftime("%b %d")
    except Exception:
        return iso[:16]


def badge_html(status: str) -> str:
    icons = {"replied": "✓", "failed": "✗", "pending": "…"}
    cls = {"replied": "badge-replied", "failed": "badge-failed"}.get(status, "badge-pending")
    return f'<span class="badge {cls}">{icons.get(status,"·")} {status}</span>'


def channel_dot(ch: str) -> str:
    color = CHANNEL_META.get(ch, {}).get("color", "#6B7280")
    return f'<span class="ch-dot" style="background:{color}"></span>'


def render_card(m: dict):
    ch = m["channel"]
    meta = CHANNEL_META.get(ch, {"label": ch, "color": "#6B7280", "icon": "💬"})
    color = meta["color"]

    subject_html = ""
    if m.get("subject"):
        subject_html = f'<div class="msg-subject">Re: {m["subject"]}</div>'

    reply_html = ""
    if m.get("reply"):
        reply_html = f'''
        <div class="bot-reply">
            <div class="bot-reply-label">Bot reply</div>
            {m["reply"]}
        </div>'''

    st.markdown(
        f'<div class="msg-card" style="border-left-color:{color}">'
        f'  <div class="msg-header">'
        f'    {channel_dot(ch)}'
        f'    <span class="sender">{m["sender"]}</span>'
        f'    {badge_html(m["status"])}'
        f'    <span class="ts">{fmt_time(m["ts"])}</span>'
        f'  </div>'
        f'  {subject_html}'
        f'  <div class="msg-body">{m["body"]}</div>'
        f'  {reply_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if m["status"] in ("pending", "failed"):
        with st.expander("↩ Retry — generate & send reply"):
            draft = st.text_area(
                "Edit before sending",
                value=m.get("reply") or "",
                key=f"draft_{m['id']}",
                height=80,
            )
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if st.button("Send reply", key=f"send_{m['id']}", type="primary"):
                    msg_obj = Message(
                        channel=m["channel"],
                        sender=m["sender"],
                        body=m["body"],
                        subject=m.get("subject"),
                    )
                    try:
                        ch_obj = get_channel(m["channel"])
                        ok = ch_obj.send(msg_obj, draft)
                        store.set_reply(m["id"], draft, "replied" if ok else "failed")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with col_b:
                if st.button("🤖 Re-run agent", key=f"agent_{m['id']}"):
                    msg_obj = Message(
                        channel=m["channel"],
                        sender=m["sender"],
                        body=m["body"],
                        subject=m.get("subject"),
                    )
                    with st.spinner("Agent working…"):
                        try:
                            pipeline.process(msg_obj, record_id=m["id"])
                        except Exception as e:
                            st.error(str(e))
                    st.rerun()


def channel_tab(ch: str, msgs: list[dict], fetch_fn=None):
    meta = CHANNEL_META.get(ch, {"label": ch, "icon": "💬"})

    col_hdr, col_btn = st.columns([5, 1])
    with col_hdr:
        st.markdown(
            f'<div class="section-title">{meta["icon"]} {meta["label"]}'
            f' — {len(msgs)} message{"s" if len(msgs) != 1 else ""}</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        btn_cols = st.columns(2 if fetch_fn else 1)
        with btn_cols[0]:
            if st.button("Refresh", key=f"ref_{ch}"):
                st.rerun()
        if fetch_fn and len(btn_cols) > 1:
            with btn_cols[1]:
                if st.button("Fetch", key=f"fetch_{ch}"):
                    with st.spinner("Fetching…"):
                        fetch_fn()
                    st.rerun()

    if not msgs:
        icon = meta["icon"]
        st.markdown(
            f'<div class="empty"><div class="empty-icon">{icon}</div>'
            f'No {meta["label"]} messages yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for m in reversed(msgs):
            render_card(m)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### AutoBot")
    st.markdown('<div style="color:#6B7280;font-size:12px;margin-bottom:1.5rem;">Multi-channel AI assistant</div>', unsafe_allow_html=True)

    page = st.radio(
        "View",
        ["All messages", "WhatsApp", "Email", "Messenger", "Threads", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown('<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #E5E7EB"></div>', unsafe_allow_html=True)

    all_msgs = store.get_messages()
    for ch, meta in CHANNEL_META.items():
        count = len([m for m in all_msgs if m["channel"] == ch])
        replied = len([m for m in all_msgs if m["channel"] == ch and m["status"] == "replied"])
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;font-size:13px;">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{meta["color"]};display:inline-block;flex-shrink:0"></span>'
            f'<span style="flex:1;color:#374151">{meta["label"]}</span>'
            f'<span style="color:#9CA3AF;font-size:12px">{replied}/{count}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top:auto;padding-top:2rem;font-size:11px;color:#9CA3AF">Auto-refreshing every 5s</div>', unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
all_msgs = store.get_messages()
wa_msgs  = [m for m in all_msgs if m["channel"] == "whatsapp"]
em_msgs  = [m for m in all_msgs if m["channel"] == "email"]
ms_msgs  = [m for m in all_msgs if m["channel"] == "messenger"]
th_msgs  = [m for m in all_msgs if m["channel"] == "threads"]

if page == "All messages":
    st.markdown(
        '<div class="topbar">'
        '<div><div class="topbar-title">All messages</div>'
        '<div class="topbar-sub">Every channel in one view</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Stat pills
    total    = len(all_msgs)
    replied  = len([m for m in all_msgs if m["status"] == "replied"])
    failed   = len([m for m in all_msgs if m["status"] == "failed"])
    pending  = len([m for m in all_msgs if m["status"] == "pending"])

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-pill"><div class="num">{total}</div><div class="lbl">Total</div></div>'
        f'<div class="stat-pill"><div class="num" style="color:#166534">{replied}</div><div class="lbl">Replied</div></div>'
        f'<div class="stat-pill"><div class="num" style="color:#854D0E">{pending}</div><div class="lbl">Pending</div></div>'
        f'<div class="stat-pill"><div class="num" style="color:#991B1B">{failed}</div><div class="lbl">Failed</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not all_msgs:
        st.markdown('<div class="empty"><div class="empty-icon">📭</div>No messages yet across any channel.</div>', unsafe_allow_html=True)
    else:
        for m in sorted(all_msgs, key=lambda x: x["ts"], reverse=True):
            render_card(m)

elif page == "WhatsApp":
    st.markdown('<div class="topbar"><div><div class="topbar-title">AutoBot · WhatsApp</div><div class="topbar-sub">Messages from +1 555 149 8140</div></div></div>', unsafe_allow_html=True)
    channel_tab("whatsapp", wa_msgs)

elif page == "Email":
    def fetch_email():
        from channels.email import EmailChannel
        ec = EmailChannel()
        for msg in ec.fetch_unread():
            pipeline.process(msg)
            ec.mark_read(msg.uid)

    st.markdown('<div class="topbar"><div><div class="topbar-title">AutoBot · Email</div><div class="topbar-sub">Gmail inbox</div></div></div>', unsafe_allow_html=True)
    channel_tab("email", em_msgs, fetch_fn=fetch_email)

elif page == "Messenger":
    st.markdown('<div class="topbar"><div><div class="topbar-title">AutoBot · Messenger</div><div class="topbar-sub">Facebook Page messages</div></div></div>', unsafe_allow_html=True)
    channel_tab("messenger", ms_msgs)

    with st.expander("Setup guide — Messenger"):
        st.markdown("""
**To connect Messenger:**

1. Go to [Meta Developer Console](https://developers.facebook.com) → your App → **Add Product → Messenger**
2. Under **Access Tokens**, select your Facebook Page and generate a token
3. Add to `.env`: `MESSENGER_PAGE_TOKEN=your_token`
4. Under **Webhooks**, set:
   - Callback URL: `https://your-ngrok.ngrok.io/messenger-webhook`
   - Verify Token: same as `MESSENGER_WEBHOOK_TOKEN` in `.env`
5. Subscribe to **messages** field
6. Restart the bot: `docker compose restart bot`
        """)

elif page == "Threads":
    st.markdown('<div class="topbar"><div><div class="topbar-title">AutoBot · Threads</div><div class="topbar-sub">Meta Threads API</div></div></div>', unsafe_allow_html=True)
    channel_tab("threads", th_msgs)

    with st.expander("Setup guide — Threads"):
        st.markdown("""
**To connect Threads:**

1. Go to [Meta Developer Console](https://developers.facebook.com) → your App → **Add Product → Threads API**
2. Add your Instagram account and complete the review
3. Generate a long-lived User Access Token with permissions:
   - `threads_basic`
   - `threads_content_publish`
   - `threads_manage_replies`
4. Add to `.env`:
   - `THREADS_ACCESS_TOKEN=your_token`
   - `THREADS_USER_ID=your_numeric_user_id`
5. Restart: `docker compose restart bot`

> ⚠️ Threads does not yet support incoming DM webhooks. Replies can be triggered manually from this dashboard.
        """)

elif page == "Settings":
    st.markdown('<div class="topbar"><div><div class="topbar-title">AutoBot · Settings</div><div class="topbar-sub">Configuration & environment</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Channel status</div>', unsafe_allow_html=True)

    env_checks = {
        "WhatsApp":  ["WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_ID"],
        "Email":     ["GMAIL_USER", "GMAIL_APP_PASSWORD"],
        "Messenger": ["MESSENGER_PAGE_TOKEN"],
        "Threads":   ["THREADS_ACCESS_TOKEN", "THREADS_USER_ID"],
    }
    for ch_label, keys in env_checks.items():
        all_set = all(os.getenv(k) for k in keys)
        status = "✓ Connected" if all_set else "✗ Not configured"
        color  = "#166534" if all_set else "#9CA3AF"
        missing = [k for k in keys if not os.getenv(k)]
        hint = "" if all_set else f" — add {', '.join(missing)} to .env"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid #F3F4F6;">'
            f'<span style="font-weight:500;color:#374151;width:100px">{ch_label}</span>'
            f'<span style="color:{color};font-size:13px">{status}{hint}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<br><div class="section-title">LLM</div>', unsafe_allow_html=True)
    llm_url = os.getenv("LLM_URL", "not set")
    llm_model = os.getenv("LLM_MODEL", "not set")
    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:13px;'
        f'background:#F8FAFC;border:1px solid #E2E8F0;border-radius:7px;padding:12px 14px;color:#374151;">'
        f'URL &nbsp;&nbsp; {llm_url}<br>'
        f'Model &nbsp;{llm_model}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<br><div class="section-title">Webhook URLs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:13px;'
        'background:#F8FAFC;border:1px solid #E2E8F0;border-radius:7px;padding:12px 14px;color:#374151;">'
        'WhatsApp &nbsp;&nbsp; POST /webhook<br>'
        'Messenger &nbsp; POST /messenger-webhook<br>'
        'Health &nbsp;&nbsp;&nbsp;&nbsp; GET  /health'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(5)
st.rerun()
