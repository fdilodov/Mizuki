# Mizu (水) - ki (木) AI Assistant

Mizu is a RAG (Retrieval-Augmented Generation) chatbot designed to assist researchers at the Hyper-Kamiokande experiment. It uses **Groq (Llama 3.3)** for fast reasoning and **ChromaDB** for local document storage.

## 🚀 Features
* **Specialised Knowledge:** Answers questions based on internal documents.
* **Privacy Focused:** Documents are processed locally; only anonymised text chunks are sent to the LLM.
* **Citations:** Every answer includes the source document filename and page number.
* **Math & Code:** Supports LaTeX physics formulas ($E=mc^2$) and Python scripting.
