from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert analyst for ESG reports.

Use the context below to answer the question. 
Always cite the page number(s) in parentheses after each fact, e.g., "(Page 3)".
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)