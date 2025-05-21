import streamlit as st
from openai import OpenAI
import PyPDF2
import io
import re
import time
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Sentiment Analysis of WhatsApp Chat (with AI/ML)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
defaults = {
    "chat_history": [{"role": "system", "content": "You are a helpful assistant powered by LLaMA 3 on Groq."}],
    "file_confirmation": "",
    "last_uploaded": None,
    "uploaded_file_text": None,
    "file_type": None,
    "whatsapp_df": None,
    "awaiting_response": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Set API key from secrets
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = st.secrets.get("groq_api_key", "")

def parse_whatsapp_to_df(file_text):
    pattern = r"(\d{2}/\d{2}/\d{4}), (\d{1,2}:\d{2}\s?[APMapm]{2}) - (.*?): (.*)"
    rows = []
    for line in file_text.splitlines():
        match = re.match(pattern, line)
        if match:
            date, time_, sender, message = match.groups()
            rows.append({"date": date, "time": time_, "sender": sender, "message": message})
    return pd.DataFrame(rows) if rows else None

with st.sidebar:
    st.header("Chat Controls")

    if st.button("Clear Chat"):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file (WhatsApp chat supported)", type=["pdf", "txt"])

    if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded:
        file_text = ""
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_text += page.extract_text() or ""
        elif uploaded_file.type == "text/plain":
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_text = stringio.read()

        st.session_state.uploaded_file_text = file_text
        st.session_state.last_uploaded = uploaded_file

        wa_line_found = any(
            re.match(r"\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s?[APMapm]{2} - .+: .+", line)
            for line in file_text.splitlines()
        )
        if wa_line_found:
            st.session_state.file_type = "whatsapp"
            st.session_state.file_confirmation = "WhatsApp chat uploaded! You can now ask anything about it, including bar graphs."
            st.session_state.whatsapp_df = parse_whatsapp_to_df(file_text)
        elif file_text.strip():
            st.session_state.file_type = "generic"
            st.session_state.file_confirmation = "Document uploaded! You can now ask anything about it."
            st.session_state.whatsapp_df = None
        else:
            st.session_state.file_confirmation = "Could not extract content from the file."
            st.session_state.whatsapp_df = None

    if st.session_state.file_confirmation:
        st.success(st.session_state.file_confirmation)

st.title("Sentiment Analysis of WhatsApp Chat (with AI/ML)")

def render_chat():
    chat_to_display = st.session_state.chat_history
    if st.session_state.awaiting_response and chat_to_display[-1]["role"] == "user":
        for msg in chat_to_display[:-1]:
            render_message(msg)
        render_message(chat_to_display[-1])
        st.info("Assistant is typing...")
    else:
        for msg in chat_to_display:
            render_message(msg)

def render_message(msg):
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='max-width: 70%; background-color: #d4edda; padding: 10px 14px; border-radius: 12px;'>
                    <div style='font-weight: bold; margin-bottom: 4px; color: #155724;'>You</div>
                    <div style='word-wrap: break-word;'>{msg['content']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif msg["role"] == "assistant":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <div style='max-width: 70%; background-color: #f1f0f0; padding: 10px 14px; border-radius: 12px;'>
                    <div style='font-weight: bold; margin-bottom: 4px; color: #333;'>Assistant</div>
                    <div style='word-wrap: break-word;'>{msg['content']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("### Chat")
render_chat()

user_input = st.chat_input("Type your message here...")

def plot_and_download_bar(df, x, y, title, xlabel, ylabel, fname="bar.png"):
    fig, ax = plt.subplots()
    df.plot(kind="bar", x=x, y=y, legend=False, ax=ax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download bar graph as PNG", buf.getvalue(), file_name=fname, mime="image/png")
    plt.close(fig)

if user_input and not st.session_state.awaiting_response:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.awaiting_response = True
    st.rerun()

if st.session_state.awaiting_response and st.session_state.chat_history[-1]["role"] == "user":
    chat = st.session_state.chat_history.copy()
    context_text = ""

    if st.session_state.file_type == "whatsapp" and st.session_state.whatsapp_df is not None:
        msg = st.session_state.chat_history[-1]["content"].lower()
        df = st.session_state.whatsapp_df

        if "bar graph" in msg and ("participant" in msg or "user" in msg or "sender" in msg):
            counts = df["sender"].value_counts().reset_index()
            counts.columns = ["Participant", "Message Count"]
            st.markdown("#### Bar graph: Messages per participant")
            plot_and_download_bar(counts, "Participant", "Message Count", "Messages per Participant", "Participant", "Messages")
            st.session_state.chat_history.append({"role": "assistant", "content": "Here is the bar graph of messages per participant."})
            st.session_state.awaiting_response = False
            st.rerun()
        elif "bar graph" in msg and ("date" in msg or "day" in msg):
            counts = df["date"].value_counts().sort_index().reset_index()
            counts.columns = ["Date", "Message Count"]
            st.markdown("#### Bar graph: Messages per day")
            plot_and_download_bar(counts, "Date", "Message Count", "Messages per Day", "Date", "Messages")
            st.session_state.chat_history.append({"role": "assistant", "content": "Here is the bar graph of messages per day."})
            st.session_state.awaiting_response = False
            st.rerun()
        else:
            context_text = f"The following WhatsApp chat has been uploaded. Use it as context:\n{st.session_state.uploaded_file_text[:15000]}"
    elif st.session_state.uploaded_file_text:
        context_text = f"The following document has been uploaded. Use it as context:\n{st.session_state.uploaded_file_text[:15000]}"

    if context_text:
        chat.append({"role": "system", "content": context_text})

    with st.spinner("Assistant is typing..."):
        try:
            client = OpenAI(api_key=st.session_state.groq_api_key, base_url="https://api.groq.com/openai/v1")
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=chat,
                stream=True
            )
            assistant_response = ""
            for chunk in response:
                assistant_response += chunk.choices[0].delta.content or ""
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

        st.session_state.awaiting_response = False
        st.rerun()
