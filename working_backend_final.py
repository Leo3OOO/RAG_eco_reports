import os
import json
from dotenv import load_dotenv
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
# environment configurations
# suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# allow duplicate OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import RAG and embedding libraries
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.utils import filter_complex_metadata

# OpenAI client
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

import time
from langchain_community.document_loaders import PyPDFLoader

# load pdf and split 
def read_and_split_pdf(file_path, chunk_size=2000, chunk_overlap=200):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import psutil, time

    # loads pdf using pypdf loader (docling caused Memory problems)
    print("📄 Loading PDF (lightweight)...")
    start = time.time()
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # splitting 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # chunking in order increase speed and reduce RAM burdon
    print(f"📦 Splitting PDF into chunks of {chunk_size} characters with {chunk_overlap} overlap...")
    texts = splitter.split_documents(documents)
    
    # provides user information about compute time and RAM (we had some issues with this, thats why we were quite thorough in this step)
    print(f"📦 Chunks: {len(texts)}")
    print(f"🧠 RAM usage: {psutil.virtual_memory().used / 1e9:.2f} GB")
    print(f"⏱️ Time: {time.time() - start:.2f} seconds")
    return texts

# Embeddings class for AcademicCloud API
class APIEmbeddings(Embeddings):
    def __init__(self, api_key, api_base="https://chat-ai.academiccloud.de/v1", model="e5-mistral-7b-instruct", batch_size=32):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.batch_size = batch_size
    def embed_documents(self, texts):
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = []
        total = len(texts)
        print(f"Embedding {total} chunks in batches of {self.batch_size}...")
        start = time.time()
        for i in range(0, total, self.batch_size):
            batch = texts[i:i+self.batch_size]
            print(f"Embedding batch {i//self.batch_size+1} ({i+1}-{min(i+self.batch_size, total)})...")
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                encoding_format="float"
            )
            embeddings.extend([d.embedding for d in response.data])
        print(f"Embedding done in {time.time()-start:.1f} seconds.")
        return embeddings
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# local Embeddings class for AcademicCloud API
# we can switch between local and cloud embeddings in case the computer is too slow
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    def embed_documents(self, texts):
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = []
        total = len(texts)
        print(f"Embedding {total} chunks locally in batches of {self.batch_size}...")
        for i in range(0, total, self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_emb = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_emb)
        return embeddings
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# vectorize texts and store in chroma db
def embed_and_store(texts, index_path=None):
    load_dotenv() # loads API key incase not yet done
    embeddings = LocalEmbeddings()
    persist_dir = index_path if index_path else "chroma_db"
    print(f"[embed_and_store] Number of texts to embed: {len(texts)}")
    if texts:
        print(f"[embed_and_store] Sample text: {texts[0].page_content[:200]}")
    
    # Filter complex metadata for Chroma compatibility
    for doc in texts:
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            doc.metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool, type(None)))}

    # Initialize empty Chroma vector store
    vector_store = Chroma(
        collection_name="default",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # adding embeddings in batches
    batch_size = embeddings.batch_size
    total = len(texts)

    print(f"[embed_and_store] Inserting in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch_docs = texts[i:i + batch_size]
        vector_store.add_documents(batch_docs)
        print(f"[embed_and_store] Added batch {i // batch_size + 1} ({i+1}-{min(i + batch_size, total)}) to Chroma DB")

    print(f"[embed_and_store] Chroma DB created at {persist_dir}")
    return vector_store

# setting retriever from vector database
def set_retriever(vector_store):
    """Return simple retriever from Chroma store"""
    return vector_store.as_retriever()

# initialising llm
def set_llm():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if api_key is not None:
        from pydantic import SecretStr
        api_key = SecretStr(api_key)
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://chat-ai.academiccloud.de/v1",
        model="meta-llama-3.1-8b-rag", # choosing model 
        temperature=0.4 # lowering temperature to reduce randomness
    )

# extracting info and creating json
def extract_info_as_json(llm, retriever, pdf_name):
    """Extract structured info from the PDF using RAG and return JSON."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # we decided to write a short prompt for each of the json keys. the idea is for the llm to give a very short answer so that the information is clearly visible
    schema_questions = {
        "CO2": "What does the document say about CO2 emissions? please be as concise as possible",
        "NOX": "What information is provided about NOX emissions? please be as concise as possible",
        "Number of Electric Vehicles": "How many electric vehicles are mentioned or estimated? please be as concise as possible",
        "Impact": "What impact is discussed in the document? please be as concise as possible",
        "Risks": "What risks are described? please be as concise as possible",
        "Opportunities": "What opportunities are mentioned? please be as concise as possible",
        "Strategy": "What strategy is outlined? please be as concise as possible",
        "Actions": "What actions are proposed or taken? please be as concise as possible",
        "Adopted_policies": "What policies have been adopted? please be as concise as possible",
        "Targets": "What targets are defined in the document? please be as concise as possible"
    }

    result = {"name": pdf_name}
    for key, question in schema_questions.items():
        try:
            answer = qa_chain.run(question)
            result[key] = answer.strip()
        except Exception as e:
            result[key] = f"Error: {str(e)}"
    return result

# creating chat loop and conversational awareness
def chat_loop(client, retriever):
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant. Use the provided PDF to answer the user's questions."
        )
    ]
    print("Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.strip().lower() == "exit":
            break
        context = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(question))
        messages.append(ChatCompletionSystemMessageParam(role="system", content=f"CONTEXT:\n{context}"))
        messages.append(ChatCompletionUserMessageParam(role="user", content=question))
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="meta-llama-3.1-8b-rag",
            temperature=0.4
        )
        answer = chat_completion.choices[0].message.content or ""
        messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=answer))
        print(f"AI: {answer}")
