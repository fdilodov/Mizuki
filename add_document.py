#!/usr/bin/env python
# Script to add new file to the already existing chroma_db
# To run it: python add_document.py ./documents/new_file.pdf
#
import os
import sys
import re
import argparse
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- 1. Reuse the exact same cleaning logic ---
def clean_text(text):
    """
    Cleaning logic to remove PDF artifacts like line numbers.
    Must match ingest.py to ensure data consistency.
    """
    text = re.sub(r'^\s*\d+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def add_single_file(file_path):
    # 1. Verification
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found at {file_path}")
        return
    
    if not os.path.exists("chroma_db"):
        print("[-] Error: 'chroma_db' directory not found. Please run ingest.py first to create the database.")
        return

    print(f"[+] Processing file: {file_path}")

    # 2. Load and Clean (Same settings as ingest.py)
    try:
        loader = UnstructuredPDFLoader(
            file_path=file_path,
            mode="elements",
            strategy="hi_res",
            languages=["eng", "jpn"]
        )
        raw_docs = loader.load()
        
        cleaned_docs = []
        for doc in raw_docs:
            original_text = doc.page_content
            cleaned_text = clean_text(original_text)
            if len(cleaned_text) > 0:
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)
        
        print(f"[+] Loaded {len(raw_docs)} elements -> kept {len(cleaned_docs)} after cleaning.")
        
    except Exception as e:
        print(f"[-] Failed to load file: {e}")
        return

    # 3. Split (Same settings as ingest.py)
    # Note: ingest.py uses chunk_size=10000. This is very large, but we keep it to match your DB.
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    text_chunks = textSplitter.split_documents(cleaned_docs)
    cleaned_chunks = filter_complex_metadata(text_chunks)
    
    if not cleaned_chunks:
        print("[-] No valid text chunks found to add.")
        return

    print(f"[+] Prepared {len(cleaned_chunks)} chunks for addition.")

    # 4. Append to Existing DB
    # We initialize Chroma pointing to the SAME persist_directory
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embedding
    )
    
    print("[+] Appending to vector store...")
    vectorstore.add_documents(cleaned_chunks)
    print(f"[+] Success! Added {file_path} to the database.")

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Add a single PDF to the existing Vector DB.")
    parser.add_argument("filepath", help="Path to the PDF file you want to add")
    args = parser.parse_args()

    add_single_file(args.filepath)
