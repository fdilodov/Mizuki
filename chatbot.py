import os
import sys
import re
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- LLM SETUP ---
def setup_llm():
    """
    Hybrid Setup:
    1. If GROQ_API_KEY exists -> Use Groq (Cloud, Fast, llama-3.3-70b-versatile).
    2. Else -> Use Local Llama (Mac, Offline).
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        print("[+] Detected Cloud Environment. Using Groq (Llama 3).")
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0.0,
            model_name="llama-3.3-70b-versatile", # 70B is excellent for Math/Coding
            api_key=groq_api_key
        )
    else:
        print("[!] No API Key found. Using Local Llama (Mac Mode).")
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(
            streaming=True,
            model_path="./models/llama-pro-8b-instruct.Q8_0.gguf",
            temperature=0.0, 
            top_p=0.1,
            n_batch=2048,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

# --- EMBEDDINGS ---
def setup_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    # Must match the model used in ingest.py
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="chroma_db", embedding_function=embedding)

# --- HELPER FUNCTIONS ---
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_number", "?")
        formatted.append(f"SOURCE: {os.path.basename(source)} (Page {page})\nCONTENT: {doc.page_content}")
    return "\n\n".join(formatted)

def find_best_sentence(text, query):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    query_words = set(query.lower().split())
    best_sentence = text[:200]
    max_overlap = -1
    for sentence in sentences:
        overlap = len(query_words.intersection(set(sentence.lower().split())))
        if overlap > max_overlap and len(sentence) > 10:
            max_overlap = overlap
            best_sentence = sentence.strip()
    return best_sentence

# --- CHATBOT LOGIC ---
if __name__ == "__main__":
    # This block is for testing chatbot.py directly in terminal
    llm = setup_llm()
    print("Chatbot CLI Ready. Type 'q' to exit.")
    while True:
        q = input(">> ")
        if q in ['q', 'exit']: break
        print(llm.invoke(q).content)
        
