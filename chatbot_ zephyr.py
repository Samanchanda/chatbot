

import os
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Set page config
st.set_page_config(page_title="💬 Smart AI Chat", layout="centered")


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_Api"

# Load model
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="conversational"
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_model()

# Function to clean unwanted tags
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


st.markdown("<h2 style='color:#FF69B4;'>🌸 Saman's Sweet ChatBot 💖</h2>", unsafe_allow_html=True)


st.markdown("Chat with a smart assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]
    st.session_state.messages = []

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user's message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.messages.append(("You", user_input))

    # Get response from model
    response = chat_model.invoke(st.session_state.chat_history)
    cleaned = clean_response(response.content)

    # Add AI response
    st.session_state.chat_history.append(AIMessage(content=cleaned))
    st.session_state.messages.append(("AI", cleaned))

# Display messages
for sender, msg in st.session_state.messages:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)
