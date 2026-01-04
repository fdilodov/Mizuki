#!/usr/bin/env python
import streamlit as st
import pandas as pd
import os
import re
# Import your existing setup logic
from chatbot import setup_embeddings, setup_llm, format_docs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & PERSONA ---
st.set_page_config(
    page_title="Mizu (水) - Hyper-Kamiokande Assistant", 
    page_icon="./images/mizu.png",
    layout="wide"  
)

# Layout: Hide the Streamlit Header, Main Menu, and Footer
st.markdown("""
    <style>
        /* Hides the main menu (three dots in top right) */
        #MainMenu {visibility: hidden;}
        
        /* Hides the "Deploy" button */
        .stDeployButton {display:none;}
        
        /* Hides the footer (Made with Streamlit) */
        footer {visibility: hidden;}
        
        /* Hides the entire top header bar (where the GitHub icon lives) */
        header {visibility: hidden;}
        [data-testid="stHeader"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Layout: Logo + Title
col1, col2 = st.columns([0.05, 0.95], vertical_alignment="center")
with col1:
    st.image("./images/mizuhkai_logo.png", width=100)
with col2:
    st.title("Mizu (水): Hyper-Kamiokande AI")

# --- INTRODUCTION SECTION ---
st.markdown("### Meet the Mizu Research Assistants")
st.write("**Konnichiwa! I am Mizu (水), your research assistant for the Hyper-Kamiokande experiment.** There are different Mizu research assistants for the Hyper-Kamiokande experiment:")

# Row 1: Mizu Doc (Neon Blue)
c1, c2 = st.columns([0.03, 0.97], vertical_alignment="center")
with c1:
    st.image("./images/mizu_doc.png", width=40)
with c2:
    st.markdown("**Mizu Doc:** For documents and general documentation.")

# Row 2: Mizu Tech (Irish Green)
c1, c2 = st.columns([0.03, 0.97], vertical_alignment="center")
with c1:
    st.image("./images/mizu_tech.png", width=40)
with c2:
    st.markdown("**Mizu Tech:** For detector construction and operation.")

# Row 3: Mizu Soft (Silver)
c1, c2 = st.columns([0.03, 0.97], vertical_alignment="center")
with c1:
    st.image("./images/mizu_soft.png", width=40)
with c2:
    st.markdown("**Mizu Soft:** For software and computing queries.")

# Row 4: Mizu Phys (Old Gold)
c1, c2 = st.columns([0.03, 0.97], vertical_alignment="center")
with c1:
    st.image("./images/mizu_phys.png", width=40)
with c2:
    st.markdown("**Mizu Phys:** For physics parameters and analysis.")

st.markdown("---") 
# ------------------------------------

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I am ready to assist with the next generation of neutrino detection. Ask me about the Cherenkov detectors, photo-sensors, or excavation plans."
        }
    ]

# Helper Function
def find_best_sentence(text, query):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    query_words = set(query.lower().split())
    best_sentence = text[:200] + "..." 
    max_overlap = -1
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words.intersection(sentence_words))
        if overlap > max_overlap and len(sentence.strip()) > 10: 
            max_overlap = overlap
            best_sentence = sentence.strip()
    return best_sentence

# Load AI Components
@st.cache_resource
def load_resources():
    vectorstore = setup_embeddings()
    llm = setup_llm()
    return vectorstore, llm

vectorstore, llm = load_resources()

# --- SIDEBAR (Updated for Sources.txt) ---
with st.sidebar:
    st.header("System Status")
    
    # 1. Memory Check
    if os.path.exists("chroma_db"):
        st.success("Mizu Memory: Online")
    else:
        st.error("Memory Not Found. Run ingest.py first!")

    st.markdown("---")
    
    # 2. Sources Viewer (Updated to read sources.txt)
    st.header("Sources")
    
    sources_path = "./documents/sources.txt"
    
    if os.path.exists(sources_path):
        with open(sources_path, "r") as f:
            source_content = f.read()
        
        if source_content.strip():
            # Display the content as Markdown so links are clickable
            st.markdown(source_content)
        else:
            st.warning("sources.txt is empty.")
    else:
        st.info(f"File not found at: {sources_path}")

    st.markdown("---")
    
    # 3. Clear Conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Conversation cleared. How can I help with the Hyper-K project?"
            }
        ]
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    # Determine icon based on role
    if message["role"] == "assistant":
        # Default to mizu_doc for generic chat
        avatar_icon = "./images/mizu_doc.png" 
    else:
        avatar_icon = "./images/scientist.png" 

    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Verified Sources"):
                st.markdown(message["sources"])

# --- CHAT INPUT LOGIC ---
if prompt_input := st.chat_input("Ask Mizu about PMT modules or water systems..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    with st.chat_message("user", avatar="./images/scientist.png"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="./images/mizu_doc.png"):
        with st.spinner("Mizu is searching the archives..."):
            
            results = vectorstore.similarity_search_with_score(prompt_input, k=3)
            valid_docs = [doc for doc, score in results if (1.0 - (score/2)) >= 0.5]

            if not valid_docs:
                response = "I searched the database but couldn't find specific details on that topic within the current Hyper-K documentation."
                source_text = "No relevant documents found."
            else:
                context = format_docs(valid_docs)
                sys_prompt = ChatPromptTemplate.from_template(
                    "You are Mizu, an expert research assistant for Hyper-Kamiokande.\n"
                    "Guidelines:\n"
                    "1. Answer strictly based on the Context below.\n"
                    "2. If writing math/physics formulas, ALWAYS use LaTeX formatting (e.g., $E=mc^2$).\n"
                    "3. If writing code, use Markdown code blocks.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}"
                )
                chain = (sys_prompt | llm | StrOutputParser())
                response = chain.invoke({"context": context, "question": prompt_input})
                
                formatted_sources = []
                for doc in valid_docs:
                    # Get the full path from metadata
                    full_path = doc.metadata.get('source', 'Unknown')
                    
                    # Clean the path to show ONLY the filename
                    source_name = os.path.basename(full_path)
                    
                    page_num = doc.metadata.get('page_number', 'Unknown')
                    relevant_sentence = find_best_sentence(doc.page_content, prompt_input)
                    
                    formatted_sources.append(f"**{source_name} (Page {page_num})**\n> \"{relevant_sentence}\"")
                
                source_text = "\n\n".join(formatted_sources)

            st.markdown(response)
            if valid_docs:
                with st.expander("View Verified Sources"):
                    st.markdown(source_text)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "sources": source_text
            })

