print("hi")

# --- Cell Separator ---

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import streamlit as st


def ask_pdf(pdf_file_path, question):
    loader = PyPDFLoader(pdf_file_path)
    document = loader.load()
    print(bool(document))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(texts, embedding_model)

    retriever = vectorstore.as_retriever()
    #retrieved_docs = retriever.get_relevant_documents("What is important in policy making?")
    #print(retrieved_docs)

    load_dotenv()

    # setting up the llm using the api key from goettingen
    llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        openai_api_base="https://chat-ai.academiccloud.de/v1",
        model_name="meta-llama-3.1-8b-instruct",
        temperature=0.7
    )

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = question
    result = qa_chain({"query": query})

    print("Answer:\n", result["result"])
    print("\nSources:\n", result["source_documents"])