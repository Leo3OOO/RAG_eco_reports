import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=embed_model)
