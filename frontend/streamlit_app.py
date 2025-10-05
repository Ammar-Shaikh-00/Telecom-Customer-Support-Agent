import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="📱 Telecom Support Bot", page_icon="📡", layout="centered")

st.title("📱 Telecom Customer Support Bot")
st.write("Ask me anything about your telecom services, or click one of the quick FAQs below 👇")

# --- Quick FAQ Buttons ---
sample_faqs = [
    "How can I recharge my mobile balance?",
    "How to check remaining data balance?",
    "What should I do if my SIM is lost?",
    "How to activate international roaming?",
    "How to pay my postpaid bill?",
    "How can I share balance with another number?",
    "How to register a complaint?",
    "What should I do if my internet is too slow?",
]

cols = st.columns(2)
for i, faq in enumerate(sample_faqs):
    if cols[i % 2].button(faq):
        st.session_state["last_question"] = faq

# --- User Input ---
user_question = st.text_input("💬 Type your question here:", st.session_state.get("last_question", ""))

if st.button("Ask") and user_question.strip():
    try:
        response = requests.post(API_URL, json={"question": user_question})
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.success(answer)
        else:
            st.error("⚠️ Something went wrong. Try again.")
    except Exception as e:
        st.error(f"❌ Could not reach backend: {e}")
