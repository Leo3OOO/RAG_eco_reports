import streamlit as st
from test_rag_backend_v2_combined import ask_pdf
import tempfile

st.set_page_config(page_title="Ask Your PDF", layout="wide")
st.title("📄💬 Ask Questions About Your PDF")

st.markdown("""
This app allows you to upload a PDF and ask any question about its content.
The system will read the document and give an intelligent response using vector search and a language model.
""")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    question = st.text_input("What would you like to know?", placeholder="e.g. What is the main conclusion of this report?")

    if question:
        with st.spinner("Analyzing document and retrieving answer..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                answer = ask_pdf(tmp_path, question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")