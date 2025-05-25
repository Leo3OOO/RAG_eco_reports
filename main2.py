import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["HF_HOME"] = "C:/Users/StrietzelLeo/hf_cache"
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling_parser import DoclingStreamBlobParser
from embedding_utils import get_embedding_model

# --- Streamlit Page Config ---
st.set_page_config(page_title="Docling RAG Streamlit App", layout="wide")
st.title("📄 Docling RAG with Streamlit")

embedding = get_embedding_model()

# --- PDF Upload & RAG Pipeline ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    st.info(f"Processing file: {filename}")
    st.write("Reading file bytes...")
    # Blob is imported from docling_parser.py
    from docling_parser import Blob
    blob = Blob(data=file_bytes, metadata={"filename": filename})
    st.write("Instantiating parser...")
    parser = DoclingStreamBlobParser(document_name=filename)
    st.write("Parsing document...")
    docs = list(parser.lazy_parse(blob))
    st.success(f"Converted to {len(docs)} document(s)")
    st.write(f"First doc preview: {docs[0].page_content[:300]}" if docs else "No docs parsed.")
    st.write("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    st.success(f"Split into {len(chunks)} chunks")
    st.write(f"First chunk preview: {chunks[0].page_content[:300]}" if chunks else "No chunks created.")
    st.write("Creating vectorstore...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding,
    )
    st.success("Vector store populated")
    st.session_state["vectorstore"] = vectorstore
    st.session_state["chunks"] = chunks
    st.session_state["doc_filename"] = filename

if "vectorstore" in st.session_state:
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Retrieving relevant chunks..."):
            results = st.session_state["vectorstore"].similarity_search(query, k=3)
        for i, doc in enumerate(results):
            st.markdown(f"**Result {i+1}:** \n{doc.page_content[:500]}...")
            st.write(f"_Source: {doc.metadata.get('source', '')}_")