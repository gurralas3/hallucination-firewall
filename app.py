import streamlit as st
import os
import sys
import html as html_lib
import pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from data_processor import process_and_index, normalize_name, DATASET_PATH
from setup_domains import setup_all
from firewall import stream_llm_answer, verify_answer, _load_cache
from gemini_client import active_backend

st.set_page_config(
    page_title="Hallucination Firewall",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #09090f; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a12 100%);
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a0533 0%, #0d1b4b 50%, #0a2a1a 100%);
    border: 1px solid #2d1b6e;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(16,185,129,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 8px;
    margin-top: 12px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Status badges ── */
.badge-corrected {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.4);
    color: #fca5a5;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-verified {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
    border: 1px solid rgba(16,185,129,0.4);
    color: #6ee7b7;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-fallback {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.35);
    color: #fcd34d;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Answer cards ── */
.answer-card {
    background: linear-gradient(135deg, #0f1a2e, #0a1220);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.answer-card-wrong {
    background: linear-gradient(135deg, #1a0a0a, #200f0f);
    border: 1px solid #4a1515;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #fca5a5;
    font-size: 0.9rem;
}
.wrong-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #f87171;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.citation-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 0.82rem;
    color: #a5b4fc;
    font-style: italic;
}
.confidence-bar-wrap {
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.confidence-label {
    font-size: 0.78rem;
    color: #64748b;
    white-space: nowrap;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #111827, #0f172a);
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #1e293b;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 16px 0;
}
.sidebar-brand-icon {
    font-size: 1.6rem;
}
.sidebar-brand-text {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-backend {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 0.75rem;
    color: #6ee7b7;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #475569;
    margin: 16px 0 8px 0;
}
.record-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 0.82rem;
}
.record-name { color: #e2e8f0; font-weight: 600; }
.record-detail { color: #64748b; font-size: 0.78rem; }

/* ── Firewall status banner ── */
.firewall-active {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #6ee7b7;
    text-align: center;
    font-weight: 600;
}
.firewall-inactive {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #fca5a5;
    text-align: center;
    font-weight: 600;
}

/* ── Sample question buttons ── */
.stButton > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #a5b4fc !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    padding: 6px 12px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #c7d2fe !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}

/* ── Dividers ── */
hr { border-color: #1e293b !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border-color: #1e293b !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Domain config ─────────────────────────────────────────────────────────────

DOMAINS = {
    "🏥 Hospital": {
        "org_id":      "hospital",
        "description": "Patient records — medications, doctors, diagnoses",
        "placeholder": "Ask about a patient...",
        "color":       "#06b6d4",
        "samples": [
            "What medication is Michael Thornton Md currently on?",
            "What is Matthew Hutchinson's medical condition?",
            "Who is the doctor treating Ronald Park?",
            "What were the test results for Jeff Brooks?",
        ]
    },
    "⚖️ Legal": {
        "org_id":      "legal",
        "description": "US Supreme Court cases — outcomes, justices, dates",
        "placeholder": "Ask about a Supreme Court case...",
        "color":       "#8b5cf6",
        "samples": [
            "Who was the Chief Justice in DRAPER v. UNITED STATES?",
            "What was the outcome of FEDERAL TRADE COMMISSION v. NATIONAL LEAD CO.?",
            "How many majority votes were in ANDERSON et al. v. CELEBREZZE?",
            "When was UNITED STATES v. R. F. BALL CONSTRUCTION CO. decided?",
        ]
    },
    "💰 Finance": {
        "org_id":      "finance",
        "description": "Bank client accounts — balances, account types, dates",
        "placeholder": "Ask about a client account...",
        "color":       "#10b981",
        "samples": [
            "What is Paul Watts's account balance?",
            "What type of account does Lisa Fuentes have?",
            "What is Scott Villa's account balance?",
            "When did Teresa Bell register her account?",
        ]
    },
    "🌍 Rural Clinic": {
        "org_id":      "chw",
        "description": "WHO Essential Medicines — safe doses for community health workers in low-resource settings",
        "placeholder": "Ask about a medicine or treatment...",
        "color":       "#f97316",
        "samples": [
            "What is the correct dose of Amoxicillin for a child with pneumonia?",
            "How should ORS be given to a child under 5 with diarrhoea?",
            "What is the dose of Artemether-Lumefantrine for a 20kg child with malaria?",
            "How do you prevent postpartum haemorrhage with Oxytocin?",
        ]
    }
}

# ── Ensure indexes ────────────────────────────────────────────────────────────

def ensure_indexes():
    hospital_index = os.path.join("org_indexes", "hospital", "index.faiss")
    if not os.path.exists(hospital_index):
        with st.spinner("Building hospital index... (first run only)"):
            process_and_index()
    legal_index   = os.path.join("org_indexes", "legal",   "index.faiss")
    finance_index = os.path.join("org_indexes", "finance", "index.faiss")
    if not os.path.exists(legal_index) or not os.path.exists(finance_index):
        with st.spinner("Building legal & finance indexes..."):
            setup_all()

ensure_indexes()

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_hospital_patients():
    df = pd.read_csv(DATASET_PATH)
    sample = df.sample(n=4, random_state=42)
    return [
        {"name": normalize_name(row["Name"]), "condition": row["Medical Condition"]}
        for _, row in sample.iterrows()
    ]

def _render_result(result: dict, llm_answer: str):
    status        = result.get("status", "VERIFIED")
    confidence    = result.get("confidence", 0.9)
    citation      = result.get("citation", "")
    contradiction = result.get("contradiction", "")
    pct           = int(confidence * 100)

    # Escape all dynamic content before inserting into HTML
    safe_answer        = html_lib.escape(result.get("final_answer", ""))
    safe_llm           = html_lib.escape(llm_answer)
    safe_contradiction = html_lib.escape(contradiction)
    safe_citation      = html_lib.escape(citation)

    if status == "CORRECTED":
        st.markdown(f"""
        <div class="answer-card-wrong">
            <div class="wrong-label">⚠ AI SAID (HALLUCINATION DETECTED)</div>
            <div style="margin-top:6px;">{safe_llm}</div>
            {"<div style='color:#f87171;margin-top:10px;'><strong>What was wrong:</strong> " + safe_contradiction + "</div>" if contradiction else ""}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="answer-card">
            <span class="badge-corrected">🔒 CORRECTED BY FIREWALL</span>
            <div style="margin-top:12px; color:#e2e8f0; font-size:0.96rem;">{safe_answer}</div>
            {"<div class='citation-box'>📄 " + safe_citation + "</div>" if citation else ""}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"🔒 Confidence: {pct}%")

    elif status == "VERIFIED":
        st.markdown(f"""
        <div class="answer-card">
            <span class="badge-verified">✓ VERIFIED</span>
            <div style="margin-top:12px; color:#e2e8f0; font-size:0.96rem;">{safe_answer}</div>
            {"<div class='citation-box'>📄 " + safe_citation + "</div>" if citation else ""}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"✅ Confidence: {pct}%")

    else:
        st.markdown(f"""
        <div class="answer-card">
            <span class="badge-fallback">⚠ UNVERIFIABLE</span>
            <div style="margin-top:12px; color:#e2e8f0;">{safe_answer}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("⚠️ Could not verify — please contact staff directly")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🔒</span>
        <span class="sidebar-brand-text">Hallucination Firewall</span>
    </div>
    """, unsafe_allow_html=True)

    backend = active_backend()
    st.markdown(f'<div class="sidebar-backend">⚡ {backend}</div>', unsafe_allow_html=True)

    st.markdown("---")

    domain_label = st.selectbox("Select Domain", list(DOMAINS.keys()), label_visibility="collapsed")
    domain = DOMAINS[domain_label]
    org_id = domain["org_id"]
    st.markdown(f'<div class="record-detail" style="margin-bottom:12px">{domain["description"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    firewall_on = st.toggle("Hallucination Firewall", value=True)
    if firewall_on:
        st.markdown('<div class="firewall-active">🔒 FIREWALL ACTIVE<br><span style="font-weight:400;font-size:0.75rem;opacity:0.8">All answers verified before delivery</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="firewall-inactive">⚠ FIREWALL DISABLED<br><span style="font-weight:400;font-size:0.75rem;opacity:0.8">Raw AI answers shown unverified</span></div>', unsafe_allow_html=True)

    if st.session_state.get("active_domain") != domain_label:
        st.session_state["active_domain"] = domain_label
        st.session_state["messages"] = []

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Records in system</div>', unsafe_allow_html=True)

    if org_id == "hospital":
        for p in get_hospital_patients():
            st.markdown(f"""
            <div class="record-item">
                <div class="record-name">{p['name']}</div>
                <div class="record-detail">{p['condition']}</div>
            </div>""", unsafe_allow_html=True)
    elif org_id == "finance":
        for name, atype, bal in [
            ("Paul Watts",   "Checking", "$11,219"),
            ("Lisa Fuentes", "Savings",  "$25,154"),
            ("Scott Villa",  "Savings",  "$68,008"),
            ("Teresa Bell",  "Checking", "$91,888"),
        ]:
            st.markdown(f"""
            <div class="record-item">
                <div class="record-name">{name}</div>
                <div class="record-detail">{atype} · {bal}</div>
            </div>""", unsafe_allow_html=True)
    elif org_id == "legal":
        for case, cj in [
            ("FTC v. NATIONAL LEAD CO.", "CJ Warren"),
            ("DRAPER v. UNITED STATES",  "CJ Warren"),
            ("ANDERSON v. CELEBREZZE",   "CJ Burger"),
            ("US v. R.F. BALL CONSTR.",  "CJ Warren"),
        ]:
            st.markdown(f"""
            <div class="record-item">
                <div class="record-name">{case}</div>
                <div class="record-detail">{cj}</div>
            </div>""", unsafe_allow_html=True)
    elif org_id == "chw":
        for med, detail in [
            ("Amoxicillin",            "Child pneumonia: 40mg/kg/day"),
            ("ORS",                    "Under 5: 50-100ml per loose stool"),
            ("Artemether-Lumefantrine","20kg child: 2 tablets per dose"),
            ("Paracetamol",            "Child: 10-15mg/kg every 4-6 hrs"),
            ("Oxytocin",               "PPH prevention: 10 IU IM"),
            ("Magnesium Sulfate",      "Eclampsia: 4g IV loading dose"),
        ]:
            st.markdown(f"""
            <div class="record-item">
                <div class="record-name">{med}</div>
                <div class="record-detail">{detail}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Try asking</div>', unsafe_allow_html=True)
    for q in domain["samples"]:
        if st.button(q, key=f"btn_{org_id}_{q}"):
            st.session_state["prefill"] = q
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

tab_chat, tab_audit = st.tabs(["💬  Chat", "📋  Audit Log"])

with tab_chat:
    # Hero
    fw_status = "ACTIVE 🔒" if firewall_on else "DISABLED ⚠"
    fw_color  = "#6ee7b7" if firewall_on else "#fca5a5"
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">{domain_label} AI Assistant</div>
        <div class="hero-sub">Powered by Gemma 4 · Post-generation verification against {domain["description"].split("—")[0].strip().lower()}</div>
        <div>
            <span class="hero-badge">Gemma 4</span>
            <span class="hero-badge">FAISS</span>
            <span class="hero-badge">Ollama</span>
            <span class="hero-badge" style="color:{fw_color};border-color:{fw_color};background:rgba(99,102,241,0.05)">Firewall {fw_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and firewall_on and msg.get("status") in ("CORRECTED", "FALLBACK"):
                fake_result = {
                    "status":       msg["status"],
                    "final_answer": msg["content"],
                    "confidence":   msg.get("confidence", 0.9),
                    "citation":     msg.get("citation", ""),
                    "contradiction": msg.get("contradiction", ""),
                }
                _render_result(fake_result, msg.get("original", msg["content"]))
            elif msg["role"] == "assistant" and firewall_on and msg.get("status") == "VERIFIED":
                fake_result = {
                    "status":       "VERIFIED",
                    "final_answer": msg["content"],
                    "confidence":   msg.get("confidence", 0.9),
                    "citation":     msg.get("citation", ""),
                    "contradiction": "",
                }
                _render_result(fake_result, msg["content"])
            else:
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input(domain["placeholder"])
    prefill    = st.session_state.pop("prefill", None)
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if firewall_on:
                cache     = _load_cache()
                cache_key = f"{org_id}::{user_input.strip().lower()}"

                if cache_key in cache:
                    result = cache[cache_key]
                    _render_result(result, result.get("original_answer", ""))
                    answer   = result["final_answer"]
                    status   = result["status"]
                    original = result.get("original_answer", "")

                else:
                    placeholder = st.empty()
                    placeholder.markdown("*⚡ Generating answer...*")
                    llm_parts = []
                    raw_box   = st.empty()
                    for chunk in stream_llm_answer(user_input, org_id):
                        llm_parts.append(chunk)
                        raw_box.markdown("".join(llm_parts) + "▌")
                    llm_answer = "".join(llm_parts)
                    raw_box.markdown(llm_answer)
                    placeholder.markdown("*🔍 Firewall verifying...*")

                    with st.spinner("Hallucination Firewall checking against records..."):
                        result = verify_answer(llm_answer, user_input, org_id)

                    raw_box.empty()
                    placeholder.empty()
                    _render_result(result, llm_answer)
                    status   = result["status"]
                    original = result.get("original_answer", llm_answer)
                    answer   = result["final_answer"]

            else:
                parts = []
                box   = st.empty()
                for chunk in stream_llm_answer(user_input, org_id):
                    parts.append(chunk)
                    box.markdown("".join(parts) + "▌")
                answer   = "".join(parts)
                box.markdown(answer)
                status   = "UNVERIFIED"
                original = answer
                result   = {}

        st.session_state.messages.append({
            "role":          "assistant",
            "content":       answer,
            "status":        status,
            "original":      original,
            "confidence":    result.get("confidence", 0.9) if firewall_on else 0,
            "citation":      result.get("citation", "")    if firewall_on else "",
            "contradiction": result.get("contradiction", "") if firewall_on else "",
        })

# ── Audit Log ─────────────────────────────────────────────────────────────────

with tab_audit:
    st.markdown("""
    <div class="hero" style="padding:24px 32px;">
        <div class="hero-title" style="font-size:1.5rem;">📋 Audit Log</div>
        <div class="hero-sub">Every AI answer intercepted and verified by the Hallucination Firewall</div>
    </div>
    """, unsafe_allow_html=True)

    cache = _load_cache()
    if not cache:
        st.info("No audit records yet. Ask some questions in the Chat tab.")
    else:
        rows = []
        for key, r in cache.items():
            rows.append({
                "Timestamp":       r.get("timestamp", "—"),
                "Domain":          r.get("org_id", key.split("::")[0]).title(),
                "Question":        r.get("question", "")[:80],
                "Status":          r.get("status", "—"),
                "Confidence":      f"{int(r.get('confidence', 0.9) * 100)}%",
                "Original Answer": r.get("original_answer", "")[:80],
                "Final Answer":    r.get("final_answer", "")[:80],
            })

        df = pd.DataFrame(rows).sort_values("Timestamp", ascending=False)

        total     = len(df)
        corrected = len(df[df["Status"] == "CORRECTED"])
        verified  = len(df[df["Status"] == "VERIFIED"])
        rate      = int(100 * corrected / max(total, 1))

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#818cf8">{total}</div>
            <div class="metric-label">Total Queries</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#f87171">{corrected}</div>
            <div class="metric-label">Hallucinations Caught</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#6ee7b7">{verified}</div>
            <div class="metric-label">Verified Clean</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#fcd34d">{rate}%</div>
            <div class="metric-label">Catch Rate</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        def color_status(val):
            if val == "CORRECTED":
                return "background-color: #3d1a1a; color: #fca5a5; font-weight:600"
            elif val == "VERIFIED":
                return "background-color: #1a3d1a; color: #6ee7b7; font-weight:600"
            return "color: #fcd34d"

        st.dataframe(
            df.style.applymap(color_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇ Download Audit Log (CSV)",
            csv, "audit_log.csv", "text/csv",
            use_container_width=False,
        )
