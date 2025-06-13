#!/usr/bin/env python

print("hello world")

import os
import json
from dotenv import load_dotenv
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# allow duplicate OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Import RAG and embedding libraries
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# OpenAI client
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

def read_and_split_pdf(file_path):
    loader = DoclingLoader(file_path)
    document = loader.load()
    print(f"Loaded: {bool(document)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")
    return texts


def embed_and_store(texts):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(texts, embedding=embeddings)
    return vector_store


def set_retriever(vector_store):
    """Return simple retriever from FAISS store"""
    return vector_store.as_retriever()


def set_llm():
    load_dotenv()
    return ChatOpenAI(
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base="https://chat-ai.academiccloud.de/v1",
        model_name="meta-llama-3.1-8b-rag",
        temperature=0.4
    )


def extract_info_as_json(llm, retriever, pdf_name):
    """Extract structured info from the PDF using RAG and return JSON."""
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

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


def chat_loop(client, retriever):
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant. Use the provided CONTEXT to answer the user's questions."
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


if __name__ == "__main__":
    pdf_file = "496.pdf"
    # Build vector store
    texts = read_and_split_pdf(pdf_file)
    store = embed_and_store(texts)
    retriever = set_retriever(store)
    llm = set_llm()

    #generate a random name for the output JSON file
    import secrets
    name_token = secrets.token_hex(nbytes=16)
    pdf_name = str(name_token)

    #with open(pdf_name + '.json', 'w') as f:
    #    json.dump(json_output, f)
    # Extract info and save to JSON
    extracted = extract_info_as_json(llm, retriever, pdf_file)
    with open(f"{pdf_name}.json", "w") as f:
        json.dump(extracted, f, indent=2)
    print("Extraction complete. Saved to extracted_info.json.")

    # Optional interactive chat
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://chat-ai.academiccloud.de/v1"
    )
    chat_loop(client, retriever)