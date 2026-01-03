#!/usr/bin/env python
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os

def test_query():
    # 1. Setup the same embedding model used in ingest.py
    print("[+] - Loading embedding model...")
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Connect to the existing persistent directory
    db_path = "chroma_db"
    if not os.path.exists(db_path):
        print(f"[-] - Error: {db_path} directory not found. Please run ingest.py first.")
        return

    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)
    
    # 3. Print database statistics
    count = vectorstore._collection.count()
    print(f"[+] - Vector store connected. Total chunks in DB: {count}")

    while True:
        query = input("\nEnter test query (or 'q' to quit): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break

        # 4. Perform similarity search with scores
        # Score is 'distance' (lower is better/closer)
        print(f"\n[+] - Searching for: '{query}'")
        results = vectorstore.similarity_search_with_score(query, k=3)

        if not results:
            print("[-] - No results found.")
            continue

        for i, (doc, score) in enumerate(results):
            print(f"\n--- Result {i+1} (Score/Distance: {score:.4f}) ---")
            print(f"Metadata: {doc.metadata}")
            print(f"Content Snippet: {doc.page_content[:300]}...")

if __name__ == "__main__":
    test_query()
    
