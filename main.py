import streamlit as st
import requests
import time

API_BASE_URL = "http://localhost:8000"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"

st.set_page_config(
    page_title="RAG Q&A",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.45; }
        }

        #MainMenu  { visibility: hidden; }
        footer     { visibility: hidden; }
        header     { visibility: hidden; }

        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > section,
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stChatInputContainer"],
        [data-testid="stChatInputContainer"] > div {
            background: #f5f3ff !important;
        }

        [data-testid="stBottom"] {
            background: #f5f3ff !important;
            padding: 12px 0 20px 0 !important;
            border-top: 1px solid #ddd6fe !important;
        }

        [data-testid="stChatInputContainer"] {
            background: #f5f3ff !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }

        [data-testid="stChatInput"] {
            border: none !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"] > div {
            border: 1.5px solid #ddd6fe !important;
            border-radius: 16px !important;
            background: #ffffff !important;
            box-shadow: 0 2px 12px rgba(124, 58, 237, 0.08) !important;
            overflow: hidden !important;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
        }

        [data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            border: none !important;
            color: #2d2a4a !important;
            font-size: 14px !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #a89ec9 !important;
        }
        [data-testid="stChatInput"] button {
            background: #7c3aed !important;
            border-radius: 10px !important;
            border: none !important;
        }
        [data-testid="stChatInput"] button:hover {
            background: #6d28d9 !important;
        }
        [data-testid="stChatInput"] button svg {
            fill: white !important;
        }

        .main .block-container {
            padding-top: 2.2rem;
            padding-bottom: 5rem;
            max-width: 740px;
        }

        .kb-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 13px;
            border-radius: 20px;
            background: #ede9fe;
            color: #5b21b6;
            border: 1px solid #ddd6fe;
            margin-bottom: 10px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .main-title {
            font-size: 27px;
            font-weight: 700;
            color: #1e1b4b;
            margin-bottom: 5px;
            line-height: 1.2;
        }

        .main-subtitle {
            font-size: 13px;
            color: #6d6a8a;
            margin-bottom: 0;
        }

        hr {
            border: none !important;
            border-top: 1px solid #ddd6fe !important;
            margin: 16px 0 !important;
        }

        .user-bubble-wrap {
            display: flex;
            justify-content: flex-end;
            margin: 8px 0 4px 80px;
            animation: fadeInUp 0.22s ease;
        }
        .user-bubble {
            background: #7c3aed;
            color: #f5f3ff;
            border-radius: 18px 18px 4px 18px;
            padding: 11px 16px;
            font-size: 14px;
            line-height: 1.65;
            max-width: 100%;
            word-wrap: break-word;
        }

        .ai-bubble-wrap {
            display: flex;
            justify-content: flex-start;
            margin: 4px 80px 8px 0;
            animation: fadeInUp 0.22s ease;
        }
        .ai-bubble {
            background: #ffffff;
            border: 1px solid #ddd6fe;
            border-left: 3px solid #7c3aed;
            color: #2d2a4a;
            border-radius: 4px 18px 18px 18px;
            padding: 13px 17px;
            font-size: 14px;
            line-height: 1.78;
            max-width: 100%;
            word-wrap: break-word;
        }

        .cache-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: #d1fae5;
            color: #065f46;
            padding: 2px 9px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: 700;
            border: 1px solid #a7f3d0;
            margin-top: 8px;
            letter-spacing: 0.04em;
        }

        .warning-box {
            background: #fff7ed;
            border-left: 3px solid #f97316;
            padding: 10px 15px;
            border-radius: 4px 8px 8px 4px;
            color: #9a3412;
            margin-bottom: 8px;
            font-size: 13px;
            animation: fadeInUp 0.22s ease;
        }

        .status-online {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 12px;
            padding: 6px 13px;
            border-radius: 20px;
            background: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .status-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #059669;
            animation: pulse 2s infinite;
            display: inline-block;
            flex-shrink: 0;
        }

        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessage"] > div:first-child {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: #ede9fe !important;
            border-right: 1px solid #ddd6fe !important;
        }
        [data-testid="stSidebar"] * { color: #4c1d95; }
        [data-testid="stSidebar"] h3 {
            color: #1e1b4b !important;
            font-size: 15px !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background: #ffffff !important;
            border: 1px solid #c4b5fd !important;
            color: #5b21b6 !important;
            border-radius: 9px !important;
            font-size: 13px !important;
            width: 100% !important;
            transition: all 0.18s ease;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background: #f5f3ff !important;
            border-color: #7c3aed !important;
        }

        [data-testid="stSpinner"] p {
            color: #8b7fb8 !important;
            font-size: 13px !important;
        }
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            font-size: 13px !important;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="kb-badge">🧠 Knowledge Base Active</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">RAG Question Answering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Ask anything from the knowledge base — powered by FastAPI + Ollama.</div>',
    unsafe_allow_html=True
)
st.divider()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if st.button("🔍 Check Backend Status"):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                st.markdown(
                    '<div class="status-online"><span class="status-dot"></span>Backend is running</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"Status: {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ─── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Display Previous Messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            cached = msg.get("cached", False)
            cache_html = '<br><span class="cache-badge">⚡ CACHED</span>' if cached else ""
            st.markdown(
                f'<div class="ai-bubble-wrap"><div class="ai-bubble">{msg["content"]}{cache_html}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="user-bubble-wrap"><div class="user-bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True
            )


# ─── Chat Input ─────────────────────────────────────────────────────────────────
if user_query := st.chat_input("Ask your question here..."):

    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-bubble-wrap"><div class="user-bubble">{user_query}</div></div>',
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                start_time = time.time()
                response = requests.post(
                    ASK_ENDPOINT,
                    json={"question": user_query},
                    timeout=60
                )
                elapsed = round(time.time() - start_time, 2)

                if response.status_code == 200:
                    data = response.json()
                    answer = (
                        data.get("answer")
                        or data.get("response")
                        or data.get("result")
                        or str(data)
                    )
                    cached    = data.get("cached", False)
                    guardrail = data.get("guardrail", "SAFE")

                    if guardrail == "UNSAFE":
                        st.markdown(
                            '<div class="warning-box">⚠️ This response was flagged by the safety guardrail.</div>',
                            unsafe_allow_html=True
                        )

                    cache_html = '<br><span class="cache-badge">⚡ CACHED</span>' if cached else ""
                    st.markdown(
                        f'<div class="ai-bubble-wrap"><div class="ai-bubble">{answer}{cache_html}</div></div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "cached": cached
                    })

                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment and try again.")
                elif response.status_code == 500:
                    st.error("Backend server error. Check your FastAPI logs.")
                else:
                    st.error(f"Unexpected error: HTTP {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI is running: uvicorn main:app --reload")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")