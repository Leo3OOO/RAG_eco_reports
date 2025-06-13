print("------------------------------------")
import os
from dotenv import load_dotenv
load_dotenv()
# suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# allow duplicate OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# — now import the rest of your libraries —
import transformers
# etc...

# %pip install -qU langchain-docling   
# %pip install -qU langchain-text-splitters
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import requests


def read_and_split_pdf(file_path): # works
    #read pdf
    loader = DoclingLoader(file_path)
    document = loader.load()
    print(bool(document))
    # Use a tokenizer to ensure chunks are <= 512 tokens
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Split into initial chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # find best parameters for use case
    texts = text_splitter.split_documents(document)
    # Further split any chunk > 512 tokens
    final_chunks = []
    for doc in texts:
        content = doc.page_content if hasattr(doc, 'page_content') else doc
        if not isinstance(content, str):
            content = str(content)
        tokens = count_tokens(content)
        if tokens <= 512:
            final_chunks.append(doc)
        else:
            # Split this chunk further by sentences
            import re
            sentences = re.split(r'(?<=[.!?]) +', content)
            current = ""
            for sentence in sentences:
                if count_tokens(current + sentence) > 512:
                    if current:
                        # Always create a Document-like object
                        if hasattr(doc, 'page_content'):
                            final_chunks.append(type(doc)(page_content=current))
                        else:
                            final_chunks.append(current)
                    current = sentence
                else:
                    current += (" " if current else "") + sentence
            if current:
                if hasattr(doc, 'page_content'):
                    final_chunks.append(type(doc)(page_content=current))
                else:
                    final_chunks.append(current)
    print(f"Number of chunks (<=512 tokens): {len(final_chunks)}")
    return final_chunks

# Remove unused embedding imports
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from openai import OpenAI

class APIEmbeddings(Embeddings):
    def __init__(self, api_key, api_base="https://chat-ai.academiccloud.de/v1", model="e5-mistral-7b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
    def embed_documents(self, texts):
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float"
        )
        return [d.embedding for d in response.data]
    def embed_query(self, text):
        return self.embed_documents([text])[0]

def embed_and_store(texts): #works
    api_key = os.getenv("API_KEY")
    print("[DEBUG] API_KEY:", api_key)
    if not api_key or api_key.strip() == "":
        raise ValueError("API_KEY is missing or empty. Check your .env file and environment.")
    embeddings = APIEmbeddings(api_key)

    # vector store text
    vector_store = FAISS.from_documents(texts, embedding=embeddings)
    return vector_store


#@st.cache_resource # import streamlit as st
def set_llm_and_retriever_and_qa_chain(vector_store):
    # Set the retriever
    retriever = vector_store.as_retriever()
    return retriever

vector_store = embed_and_store(read_and_split_pdf("496.pdf"))
retriever = set_llm_and_retriever_and_qa_chain(vector_store)

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

def get_context_from_retriever(retriever, question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

api_key = os.getenv("API_KEY")
base_url = "https://chat-ai.academiccloud.de/v1"
model = "meta-llama-3.1-8b-rag"
client = OpenAI(api_key=api_key, base_url=base_url)

messages: list[ChatCompletionMessageParam] = [
    ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant. Use the provided CONTEXT to answer the user's questions. If CONTEXT is empty, say you don't know.")
]

print("Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.strip().lower() == "exit":
        break
    context = get_context_from_retriever(retriever, question)
    # Insert context as a system message before the user message
    messages.append(ChatCompletionSystemMessageParam(role="system", content=f"CONTEXT:\n{context}"))
    messages.append(ChatCompletionUserMessageParam(role="user", content=question))
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.4
    )
    answer = chat_completion.choices[0].message.content or ""
    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=answer))
    print(f"AI: {answer}")

def convert_pdf_to_markdown(pdf_path, api_key):
    url = "https://chat-ai.academiccloud.de/v1/documents/convert?response_type=markdown"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    files = {"document": open(pdf_path, "rb")}
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.json().get("markdown", "")

# Example usage:
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if api_key is not None:
        api_key = api_key.strip("'")
    else:
        raise ValueError("API_KEY not found in environment variables.")
    pdf_path = "496.pdf"  # or any other PDF file
    markdown = convert_pdf_to_markdown(pdf_path, api_key)
    with open("output.md", "w") as f:
        f.write(markdown)
    print("PDF converted to markdown and saved as output.md")