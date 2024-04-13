import streamlit as st
import cohere
from pb_utils import PaperBot, Vectorstore
import os
os.environ["COHERE_API_KEY"] = "api_key"

co = cohere.Client(os.environ["COHERE_API_KEY"])

st.write("Paper Bot Chat Application")
# Add to vectorstore
vectorstore = Vectorstore(co, "https://arxiv.org/pdf/1706.03762.pdf")
bot = PaperBot(co, vectorstore)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(bot.chat_completion(st.session_state.messages[-1]["content"]))
    st.session_state.messages.append({"role": "assistant", "content": response})