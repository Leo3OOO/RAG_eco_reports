import streamlit as st
from test_rag_backend_v2_combined import ask_pdf
import tempfile
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


# Sidebar for file upload and chat history clear
st.sidebar.title("PDF Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
if st.sidebar.button("Clear chat history"):
    st.session_state['chat_history'] = []
    st.experimental_rerun()

# Main chat area
st.title("📄💬 Ask Questions About Your PDF")
st.markdown("""
This app allows you to upload a PDF and ask any question about its content.
The system will read the document and give an intelligent response using vector search and a language model.
""")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if uploaded_file is not None:
    # Display chat history as a conversation
    for chat in st.session_state['chat_history']:
        with st.chat_message("user"):
            st.markdown(chat['question'])
        with st.chat_message("assistant", avatar="assets/logo.svg"):
            st.markdown(chat['answer'])

    # Chat input at the bottom
    question = st.chat_input("What would you like to know?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Analyzing document and retrieving answer..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                answer = ask_pdf(tmp_path, question)
            except Exception as e:
                answer = f"An error occurred: {e}"
            with st.chat_message("assistant", avatar="assets/logo.svg"):
                st.markdown(answer)
            st.session_state['chat_history'].append({'question': question, 'answer': answer})
else:
    st.info("Please upload a PDF in the sidebar to start chatting.")