# from openai import OpenAI
import streamlit as st
from utils import AdvanceChatBot
import uuid


# Initialize session state for knowledge source and model
if "bot" not in st.session_state:
    st.session_state["bot"] = AdvanceChatBot()
if "model" not in st.session_state:
    st.session_state["model"] = st.session_state["bot"].load_model()
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "use_rag" not in st.session_state:
    st.session_state["use_rag"] = False


# title
st.title("💬📖 Chatbot Gemma3")

# sidebar
with st.sidebar:
    st.sidebar.title("📄 ISP FAQ RAG Settings")
    st.sidebar.markdown(    
        """
        This assistant uses **RAG (Retrieval-Augmented Generation)** powered by ISP Company FAQs.
        
        When RAG is enabled, your question will be answered based on the most relevant documents from the FAQ.

        📄 FAQ document path: `project_2/resources/ISP Company FAQ.pdf`

        Use the toggle below to enable or disable RAG.
        """
    )

    # Toggle for enabling/disabling RAG
    use_rag = st.sidebar.checkbox("🔍 Enable RAG", value=False)
    
    # reset chat
    if use_rag != st.session_state.use_rag:
        st.session_state.use_rag = use_rag
        st.session_state.model = st.session_state.bot.load_model(use_rag)
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # show alert
    if st.session_state.use_rag:
        st.success("**RAG is enabled.**\n\nYour questions will be answered using retrieved FAQ documents.")
    else:
        st.warning("**RAG is disabled.**\n\nThe model will answer based on general knowledge only.")


# show chats
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# get response
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # create thread_id for configuration memmory
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))
    st.session_state.thread_id = thread_id 
    config = {"configurable": {"thread_id": thread_id}}

    # get response from llm
    output = st.session_state.model.invoke(
        {"messages": [{"role": "user", "content": prompt}]}, 
        config=config
    )
    msg = output['messages'][-1].content

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)