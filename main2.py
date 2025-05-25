import streamlit as st
import io
from tempfile import mkdtemp
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.document_loaders.blob_loaders import Blob
from typing import Iterator, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_openai import OpenAI

# --- Streamlit Page Config ---
st.set_page_config(page_title="Docling RAG Streamlit App", layout="wide")
st.title("📄 Docling RAG with Streamlit")

# --- Chat Initialization ---
# Set OpenAI API key from Streamlit secrets
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://chat-ai.academiccloud.de/v1"
)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "meta-llama-3.1-8b-instruct"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask the LLM a question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    response = client.chat(
        model=st.session_state["openai_model"],
        messages=st.session_state.messages
    )
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.markdown(response.content)

# --- Custom Parser Definition ---
class DoclingStreamBlobParser(BaseBlobParser):
    """Parser using Docling's DocumentConverter for binary streams."""
    def __init__(self, document_name: str):
        self.document_name = document_name
        self.converter = DocumentConverter()

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as f:
            docling_source = DocumentStream(name=self.document_name, stream=f)
            result = self.converter.convert(docling_source)
            docling_doc = result.document
            content = docling_doc.export_to_markdown()
            metadata: Dict[str, Any] = getattr(docling_doc, "metadata", {}) or {}
            metadata["source"] = self.document_name
            yield Document(page_content=content, metadata=metadata)

# --- PDF Upload & RAG Pipeline ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    st.info(f"Processing file: {filename}")

    blob = Blob(data=file_bytes, metadata={"filename": filename})
    parser = DoclingStreamBlobParser(document_name=filename)
    docs = list(parser.lazy_parse(blob))
    st.success(f"Converted to {len(docs)} document(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    st.success(f"Split into {len(chunks)} chunks")

    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=embed_model)
    st.info(f"Loading embedding model: {embed_model}")

    milvus_uri = str(Path(mkdtemp()) / "streamlit_rag.db")
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="streamlit_docling_rag",
        connection_args={"uri": milvus_uri},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )
    st.success("Vector store populated")

    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Retrieving relevant chunks..."):
            results = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(results):
            st.markdown(f"**Result {i+1}:** \n{doc.page_content[:500]}...")
            st.write(f"_Source: {doc.metadata.get('source', '')}_")
