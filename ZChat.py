import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_extras.add_vertical_space import add_vertical_space

import os

st.subheader("ZChat")

st.markdown("Built by Zakaria Bouidane")

with st.sidebar:
    st.title("ZChat")
    st.subheader("This app lets you chat with ZChat! [👉]")
    api_key = st.text_input("Enter your API Key", type="password")
    add_vertical_space(2)
    st.markdown("""
    For more information 
   
    Contact me!
    """)
    add_vertical_space(3)
    st.write("Reach out to me on [LinkedIn](https://www.linkedin.com/in/zakariabouidane/)")


# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Only initialize ChatOpenAI and ConversationChain if API key is provided
if api_key:
    if st.session_state.conversation is None:
        llm = ChatOpenAI(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            openai_api_key=api_key,
            openai_api_base="https://api.fireworks.ai/inference/v1"
        )
        st.session_state.conversation = ConversationChain(
            memory=st.session_state.buffer_memory, 
            llm=llm
        )

    # Rest of your chat interface code
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.predict(input=prompt)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
else:
    st.warning("Please enter your API Key in the sidebar to start the chat.")
