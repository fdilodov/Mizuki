#!/usr/bin/env python
# Optimized ingest script with Parallel Processing and Fast Strategy
import os
import shutil
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- CONFIGURATION ---
# Set to "hi_res" only if you need to OCR images. "fast" is 100x faster.
PDF_STRATEGY = "hi_res" 
# Number of parallel processes (default to number of CPU cores)
MAX_WORKERS = os.cpu_count() 
# ---------------------

def clean_text(text):
    """Cleaning logic from your original script."""
    text = re.sub(r'^\s*\d+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_single_file(path):
    """
    Worker function to load and clean a single PDF file.
    Running this in a separate process avoids the GIL bottleneck.
    """
    if not os.path.exists(path):
        return []

    try:
        print(f"[~] Processing: {path}")
        # Using "fast" strategy by default for speed
        loader = UnstructuredPDFLoader(
            file_path=path,
            mode="elements",
            strategy=PDF_STRATEGY, 
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
        
        return cleaned_docs
        
    except Exception as e:
        print(f"[-] ERROR in file {path}: {e}")
        return []

def load_documents_parallel():
    """Load documents using multiple CPU cores."""
    
    if os.path.exists("chroma_db"):
        if os.path.exists("chroma_db.old"):
            shutil.rmtree("chroma_db.old")
        shutil.move("chroma_db", "chroma_db.old")
        print("[+] - Moved existing vector store to 'chroma_db.old'")

    input_file_path = "./documents/inputs.txt"
    if not os.path.exists(input_file_path):
        print(f"[-] Error: {input_file_path} not found.")
        return None

    with open(input_file_path, 'r') as f:
        file_paths = [
            line.strip() for line in f 
            if line.strip() and not line.strip().startswith("#")
        ]

    all_data = []
    
    # --- PARALLEL EXECUTION START ---
    print(f"[+] Starting ingestion with {MAX_WORKERS} parallel workers (Strategy: {PDF_STRATEGY})...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all file paths to the pool
        future_to_file = {executor.submit(process_single_file, path): path for path in file_paths}
        
        for future in as_completed(future_to_file):
            path = future_to_file[future]
            try:
                docs = future.result()
                if docs:
                    all_data.extend(docs)
                    print(f"[+] Finished: {path} ({len(docs)} chunks)")
            except Exception as exc:
                print(f"[-] Generator failed for {path}: {exc}")
    # --- PARALLEL EXECUTION END ---

    if not all_data:
        print("[-] No document data loaded.")
        return None

    print(f"[+] Loaded total {len(all_data)} text elements.")

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    textSplitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    text_chunks = textSplitter.split_documents(all_data)
    
    cleaned_chunks = filter_complex_metadata(text_chunks)
    print(f"[+] - Processed {len(cleaned_chunks)} chunks for vector storage")

    vectorstore = Chroma.from_documents(
        cleaned_chunks, embedding=embedding, persist_directory="chroma_db"
    )
    print("[+] - New vector store created")
    return embedding

if __name__ == "__main__":
    # Windows/MacOS safety block for multiprocessing
    multiprocessing.freeze_support() 
    load_documents_parallel()
    
