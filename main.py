import streamlit as st
import requests
import time

API_BASE_URL = "http://localhost:8000"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"

# Page Setup
st.set_page_config(
    page_title="RAG Q&A",
    layout="centered"
)

st.markdown("""
    <style>
        .response-box {
            background-color: #1e1e2e;
            border-left: 4px solid #7c3aed;
            padding: 16px 20px;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 15px;
            margin-top: 12px;
        }
        .meta-info {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 8px;
        }
        .cache-badge {
            background: #064e3b;
            color: #6ee7b7;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .warning-box {
            background: #431407;
            border-left: 4px solid #f97316;
            padding: 10px 16px;
            border-radius: 6px;
            color: #fed7aa;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# Header
st.title("RAG Question Answering")
st.markdown("Ask anything from the knowledge base. Powered by your local FastAPI + Ollama backend.")
st.divider()


# Sidebar
with st.sidebar:
    st.header("Settings")
    st.markdown(f"**API URL:** `{API_BASE_URL}`")

    if st.button("Check Backend Status"):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                st.success("Backend is running!")
            else:
                st.warning(f"Status: {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure FastAPI is running.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.markdown("**How to start backend:**")
    st.code("uvicorn main:app --reload", language="bash")
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit + FastAPI + Ollama")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(
                f'<div class="response-box">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if "meta" in msg:
                st.markdown(
                    f'<div class="meta-info">{msg["meta"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.write(msg["content"])


# Chat Input
if user_query := st.chat_input("Ask your question here..."):

    with st.chat_message("user"):
        st.write(user_query)

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

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

                # Success
                if response.status_code == 200:
                    data = response.json()

                    answer = (
                        data.get("answer")
                        or data.get("response")
                        or data.get("result")
                        or str(data)
                    )

                    server_time = response.headers.get("X-Response-Time", f"{elapsed}s")
                    cached = data.get("cached", False)
                    cache_badge = '<span class="cache-badge">CACHED</span>' if cached else ""
                    guardrail = data.get("guardrail", "SAFE")

                    if guardrail == "UNSAFE":
                        st.markdown(
                            '<div class="warning-box">This response was flagged by the safety guardrail.</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f'<div class="response-box">{answer}</div>',
                        unsafe_allow_html=True
                    )

                    meta_text = f"Response time: {server_time} | {cache_badge}" if cached else f"Response time: {server_time}"

                    st.markdown(
                        f'<div class="meta-info">{meta_text}</div>',
                        unsafe_allow_html=True
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": meta_text
                    })

                # Rate Limit
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment and try again.")

                # Server Error
                elif response.status_code == 500:
                    st.error("Backend server error. Check your FastAPI logs.")

                # Other Errors
                else:
                    st.error(f"Unexpected error: HTTP {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to backend. "
                    "Make sure FastAPI is running: uvicorn main:app --reload"
                )
            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out. The model might still be loading. "
                    "Please try again in a few seconds."
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")